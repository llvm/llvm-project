set(LLVM_TARGET_DEFINITIONS SCFTransformOps.td)
mlir_tablegen(SCFTransformOps.h.inc -gen-op-decls)
mlir_tablegen(SCFTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRSCFTransformOpsIncGen)

add_mlir_doc(SCFTransformOps SCFLoopTransformOps Dialects/ -gen-op-doc)
