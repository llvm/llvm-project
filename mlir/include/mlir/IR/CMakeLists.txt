set(LLVM_TARGET_DEFINITIONS OpAsmInterface.td)
mlir_tablegen(OpAsmInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(OpAsmInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIROpAsmInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS SymbolInterfaces.td)
mlir_tablegen(SymbolInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(SymbolInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRSymbolInterfacesIncGen)
