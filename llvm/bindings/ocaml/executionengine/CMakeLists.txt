add_ocaml_library(llvm_executionengine
  OCAML    llvm_executionengine
  OCAMLDEP llvm llvm_target
  C        executionengine_ocaml
  LLVM     ExecutionEngine MCJIT native
  PKG      ctypes)
