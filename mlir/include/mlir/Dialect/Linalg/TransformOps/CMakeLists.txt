set(LLVM_TARGET_DEFINITIONS LinalgMatchOps.td)
mlir_tablegen(LinalgMatchOps.h.inc -gen-op-decls)
mlir_tablegen(LinalgMatchOps.cpp.inc -gen-op-defs)
add_mlir_dialect_tablegen_target(MLIRLinalgMatchOpsIncGen)

set(LLVM_TARGET_DEFINITIONS LinalgTransformOps.td)
mlir_tablegen(LinalgTransformOps.h.inc -gen-op-decls)
mlir_tablegen(LinalgTransformOps.cpp.inc -gen-op-defs)
add_mlir_dialect_tablegen_target(MLIRLinalgTransformOpsIncGen)

set(LLVM_TARGET_DEFINITIONS LinalgTransformEnums.td)
mlir_tablegen(LinalgTransformOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(LinalgTransformOpsEnums.cpp.inc -gen-enum-defs)
add_mlir_dialect_tablegen_target(MLIRLinalgTransformEnumsIncGen)

add_mlir_doc(LinalgMatchOps LinalgStructuredMatchOps Dialects/ -gen-op-doc)
add_mlir_doc(LinalgTransformOps LinalgStructuredTransformOps Dialects/ -gen-op-doc)
