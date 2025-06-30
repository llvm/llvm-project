set(LLVM_TARGET_DEFINITIONS WebAssemblySSATypes.td)
mlir_tablegen(WebAssemblySSATypeConstraints.h.inc -gen-type-constraint-decls)
mlir_tablegen(WebAssemblySSATypeConstraints.cpp.inc -gen-type-constraint-defs)

set (LLVM_TARGET_DEFINITIONS WebAssemblySSAInterfaces.td)
mlir_tablegen(WebAssemblySSAInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(WebAssemblySSAInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRWebAssemblySSAInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS WebAssemblySSAOps.td)

add_mlir_dialect(WebAssemblySSAOps wasmssa)
add_mlir_doc(WebAssemblySSAOps WebAssemblySSAOps Dialects/ -gen-dialect-doc)
