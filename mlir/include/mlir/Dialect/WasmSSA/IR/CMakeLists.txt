set(LLVM_TARGET_DEFINITIONS WasmSSATypes.td)
mlir_tablegen(WasmSSATypeConstraints.h.inc -gen-type-constraint-decls)
mlir_tablegen(WasmSSATypeConstraints.cpp.inc -gen-type-constraint-defs)

set (LLVM_TARGET_DEFINITIONS WasmSSAInterfaces.td)
mlir_tablegen(WasmSSAInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(WasmSSAInterfaces.cpp.inc -gen-op-interface-defs)
add_mlir_dialect_tablegen_target(MLIRWasmSSAInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS WasmSSAOps.td)

add_mlir_dialect(WasmSSAOps wasmssa)
add_mlir_doc(WasmSSAOps WasmSSAOps Dialects/ -gen-dialect-doc -dialect wasmssa)
