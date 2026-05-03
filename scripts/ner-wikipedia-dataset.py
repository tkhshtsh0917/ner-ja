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
    dataset = load_dataset("stockmark/ner-wikipedia-dataset", split="train")
    return (dataset,)


@app.cell
def _(dataset):
    def make_dataset(row: dict) -> dict:
        text: str = str(row["text"])
        spans: list[dict] = row["entities"]

        labels: list[str] = list(dict.fromkeys(span["type"] for span in spans))

        entities = {
            label: [span["name"] for span in spans if span["type"] == label]
            for label in labels
        }

        return {"input": text, "output": {"entities": entities}}


    updated = [make_dataset(row) for row in dataset.to_list()]
    return (updated,)


@app.cell
def _(Path, json, updated):
    dir_path = Path(__file__).parent.parent / "datasets"
    file_path = dir_path / "ner-wikipedia-dataset.jsonl"

    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)

    with file_path.open(mode="w", encoding="utf-8") as f:
        for row in updated:
            f.write(f"{json.dumps(row, ensure_ascii=False)}\n")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
