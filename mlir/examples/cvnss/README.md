# CVNSS-CIR MLIR backend (experimental v0.1)

Out-of-tree C++ backend based on the repository's MLIR standalone example.
BilaScript is the intended frontend; the current input contract is textual MLIR.
This is not ClangIR, a full CVNSS4.0 codec, or a universal file converter.

## Build
Build/install LLVM+MLIR from the same repository revision, with LLVM_ENABLE_PROJECTS=mlir.
Use CMake >= 3.31, Ninja, C++17 and Python 3.
```sh
cmake -S mlir/examples/cvnss -B build-cvnss -G Ninja -DMLIR_DIR=/path/to/lib/cmake/mlir -DLLVM_DIR=/path/to/lib/cmake/llvm
cmake --build build-cvnss --target cvnss-opt
ctest --test-dir build-cvnss --output-on-failure
build-cvnss/cvnss-opt mlir/examples/cvnss/test/demo.mlir --cvnss-lower-trit
```
The tool registers the standard MLIR dialects and passes. For LLVM lowering of
the pack function, use standard arith/func conversion passes and reconcile
unrealized casts. Audit-only mapping operations must be consumed by a separate
export pass before executable lowering.

## Contract
- Base27 ID 0..26 = 13 + 9*hi + 3*mid + lo; each trit -1, 0, +1.
- Mapping records preserve external source, target, mode, version and loss policy.
  They do not perform or validate the actual linguistic mapping.
- No arbitrary file execution. Unknown file formats belong in a future opaque blob layer.
- No claim of compression or LLM improvement. Alphabet order is intentionally not assigned.

## Roadmap
1. Native compiler tests and pinned build.
2. BilaScript AST emitter; official codec and ambiguity fixtures.
3. Versioned mapping registry, provenance, byte-preserving blob adapters.
4. Typed text/document/spatial/signal/task dialects and region-based pipelines.
5. Dynamic trit ops, target-specific lowering, benchmarking and ViPickers explorer.

License: Apache-2.0 WITH LLVM-exception.
