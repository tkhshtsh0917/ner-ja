import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    from datasets import load_dataset

    return Path, json, load_dataset


@app.cell
def _(load_dataset):
    dataset = load_dataset("Meddies/meddies-pii", "japanese", split="train")
    return (dataset,)


@app.cell
def _(dataset, json):
    updated = [
        {"input": str(row["raw"]), "output": {"entities": json.loads(row["label"])}}
        for row in dataset.to_list()
    ]
    return (updated,)


@app.cell
def _(Path, json, updated):
    dir_path = Path(__file__).parent.parent / "datasets"
    file_path = dir_path / "meddies-pii.jsonl"

    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)

    with file_path.open(mode="w", encoding="utf-8") as f:
        for row in updated:
            if len(row["output"]["entities"]) > 0:
                f.write(f"{json.dumps(row, ensure_ascii=False)}\n")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
