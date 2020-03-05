set(LLVM_TARGET_DEFINITIONS CallInterfaces.td)
mlir_tablegen(CallInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(CallInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRCallOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS ControlFlowInterfaces.td)
mlir_tablegen(ControlFlowInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ControlFlowInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRControlFlowInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS InferTypeOpInterface.td)
mlir_tablegen(InferTypeOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(InferTypeOpInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRTypeInferOpInterfaceIncGen)
