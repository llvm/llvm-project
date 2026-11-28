# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Portable semantic metadata bridge; never executes imported source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

LANGS = {"bila", "c", "cpp", "javascript", "rust", "python"}
KINDS = {"function", "type", "value", "external"}
VERSION = "0.2"
MAX_SOURCE = 8 * 1024 * 1024

def digest(data):
    return hashlib.sha256(data).hexdigest()

def reference(language, uri, raw, symbol, kind):
    if language not in LANGS or kind not in KINDS:
        raise ValueError("unsupported language or semantic kind")
    if not isinstance(symbol, str) or not symbol or not isinstance(uri, str) or not uri:
        raise ValueError("nonempty symbol and URI required")
    if len(raw) > MAX_SOURCE:
        raise ValueError("source exceeds size limit")
    d = digest(raw)
    key = json.dumps([VERSION, language, uri, d, symbol, kind],
                     ensure_ascii=True, separators=(",", ":"))
    return dict(reference_id="cvnss:"+digest(key.encode()), language=language,
                source_uri=uri, source_digest=d, symbol=symbol,
                semantic_kind=kind, schema_version=VERSION)

def python_records(uri, raw):
    tree = ast.parse(raw)
    records = []
    # Only top-level definitions: no execution, imports, or semantic inference.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "type" if isinstance(node, ast.ClassDef) else "function"
            records.append(reference("python", uri, raw,
                                     f"{node.name}@{node.lineno}:{node.col_offset}", kind))
    return records

def mlir_string(value):
    # MLIR string escapes bytes, not JSON Unicode escape sequences.
    return '"' + ''.join(chr(b) if 32 <= b < 127 and b not in (34,92)
                         else "\\" + format(b, "02X")
                         for b in value.encode("utf-8")) + '"'

def emit(records):
    lines = ["module {"]
    for record in records:
        attrs = ", ".join(k+" = "+mlir_string(v) for k,v in sorted(record.items()))
        lines.append('  "cvnss.reference"() {'+attrs+'} : () -> ()')
    return "\n".join(lines+["}", ""])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("language", choices=sorted(LANGS))
    ap.add_argument("source", type=Path)
    ap.add_argument("--symbols", type=Path, help="JSON array of explicit symbol/kind records")
    ap.add_argument("--format", choices=["json", "mlir"], default="mlir")
    args = ap.parse_args()
    with args.source.open("rb") as f:
        raw = f.read(MAX_SOURCE+1)
    if len(raw)>MAX_SOURCE:
        ap.error("source exceeds size limit")
    uri = args.source.as_posix()
    if args.symbols:
        with args.symbols.open("rb") as f:
            data = f.read(MAX_SOURCE+1)
        if len(data)>MAX_SOURCE:
            ap.error("symbol manifest exceeds size limit")
        items = json.loads(data)
        if not isinstance(items, list):
            ap.error("symbols must be a JSON array")
        records = [reference(args.language, uri, raw, x["symbol"], x["kind"]) for x in items]
    elif args.language == "python":
        records = python_records(uri, raw)
    else:
        ap.error("explicit --symbols manifest required; foreign parsing not implemented")
    print(json.dumps(records, ensure_ascii=False, indent=2) if args.format=="json" else emit(records), end="\n")

if __name__ == "__main__":
    main()
