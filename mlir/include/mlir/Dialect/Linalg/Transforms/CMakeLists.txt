set(LLVM_TARGET_DEFINITIONS LinalgTransformPatterns.td)
mlir_tablegen(TestLinalgMatmulToVectorPatterns.h.inc -gen-rewriters)
mlir_tablegen(LinalgTransformPatterns.h.inc -gen-rewriters)
add_public_tablegen_target(MLIRLinalgTransformPatternsIncGen)

# Including Linalg in TableGen requires to depends on generated files
add_dependencies(MLIRLinalgTransformPatternsIncGen LinalgOdsGen)

