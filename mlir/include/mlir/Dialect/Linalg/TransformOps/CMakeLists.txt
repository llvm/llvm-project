set(LLVM_TARGET_DEFINITIONS LinalgTransformOps.td)
mlir_tablegen(LinalgTransformOps.h.inc -gen-op-decls)
mlir_tablegen(LinalgTransformOps.cpp.inc -gen-op-defs)
mlir_tablegen(LinalgTransformOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(LinalgTransformOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRLinalgTransformOpsIncGen)

add_mlir_doc(LinalgTransformOps LinalgStructuredTransformOps Dialects/ -gen-op-doc)
