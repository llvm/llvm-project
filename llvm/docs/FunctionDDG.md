# Function-scoped DDG visualizer

`FunctionDDGDotPrinterPass` (`llvm/lib/Analysis/FunctionDDGPrinter.cpp`) is the
`dot-function-ddg` pass: a function-scoped DOT printer for the Data Dependence
Graph. It is the function-scope counterpart to the loop-scoped `DDGDotPrinterPass`
(`dot-ddg`), and is intended for **human inspection** of a function's dependence
structure.

---

## 1. Why a function-scoped DDG

The loop-scoped DDG printer (`dot-ddg`) only covers code that has been recognised
as a loop nest. To visualise the dependence structure of straight-line,
cross-loop, or fully-unrolled code, a function-wide view is needed. This pass
builds a `DataDependenceGraph` for the whole function (via the existing
`DataDependenceGraph(Function&, DependenceInfo&)` constructor) and renders it
with the shared DDG `DOTGraphTraits`, so loop- and function-scope graphs look
identical. No DDG infrastructure changes are required; the pass is purely
additive.

## 2. Running it

```bash
opt -disable-output -passes=dot-function-ddg foo.ll
```

This writes one DOT file per function, named `<prefix>.<function-name>.dot`
(default prefix `ddg`). Render with Graphviz:

```bash
dot -Tpng ddg.my_function.dot -o my_function.png
```

## 3. Flags

| Flag | Effect |
|------|--------|
| `-dot-function-ddg-filename-prefix=<prefix>` | Output filename prefix; file is `<prefix>.<function-name>.dot` (default `ddg`). |
| `-dot-function-ddg-only` | Simplified rendering (the function-scope analogue of `-dot-ddg-only`): concise node labels and edge attributes. |

## 4. What the graph shows

The graph is emitted by the shared DDG `DOTGraphTraits`
(`llvm/lib/Analysis/DDGPrinter.cpp`), identical to the loop-scoped `dot-ddg`
output but built over the whole function:

- one node per DDG node (each carrying its instruction text);
- **register def-use** edges (blue) and **memory-dependence** edges (red),
  labelled with the edge kind (or, in verbose mode, the detailed dependence);
- pi-blocks and the synthetic root node handled exactly as in the loop printer.

## 5. Tests

`llvm/test/Analysis/DDG/print-dot-function-ddg.ll` exercises the printer in both
the default and `-dot-function-ddg-only` forms, covering register def-use edges
and a memory dependence between a load and an aliasing store.

Run it with `llvm-lit`:

```bash
llvm-lit -v llvm/test/Analysis/DDG/print-dot-function-ddg.ll
```

See also the `dot-function-ddg` entry in {doc}`Passes`.
