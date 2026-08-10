# tree-sitter-mlir

[tree-sitter](https://github.com/tree-sitter/tree-sitter) grammar for MLIR
following the [lang-ref](https://mlir.llvm.org/docs/LangRef/). The parser is
incomplete, and the bench statistics on the test files in the MLIR tree are as
follows:

```
Math, 100% passed
Builtin, 100% passed
Func, 100% passed
ControlFlow, 100% passed
Tensor, 93.33% passed
Arith, 83.33% passed
SCF, 88% passed
Affine, 73.08% passed
Linalg, 51.11% passed
```
