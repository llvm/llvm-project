# Minmal MLIR binaries

This folder contains example of minimal MLIR setup that can showcase the
intended binary footprint of the framework.

- mlir-cat: ~2MB
  This includes the Core IR, the builtin dialect, the textual parser/printer,
  the support for bytecode serialization.
- mlir-minimal-opt: ~3MB
  This adds all the tooling for an mlir-opt tool: the pass infrastructure
  and all the instrumentation associated with it.
- mlir-miminal-opt-canonicalize: ~4.8MB
  This add the canonicalizer pass, which pulls in all the pattern/rewrite
  machinery, including the PDL compiler and intepreter.