set(LIBS

  MLIRAffineOps
  MLIRAnalysis
  MLIREDSC
  MLIRIR
  MLIRLinalgAnalysis
  MLIRLinalgEDSC
  MLIRLinalgOps
  MLIRLinalgUtils
  MLIRLoopOps
  MLIRPass
  MLIRStandardOps
  MLIRStandardToLLVM
  MLIRTransformUtils
  MLIRVectorOps
  )

add_mlir_dialect_library(MLIRLinalgTransforms
  Fusion.cpp
  LinalgTransforms.cpp
  LinalgToLoops.cpp
  Promotion.cpp
  Tiling.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg
  DEPENDS
  intrinsics_gen
  )

add_dependencies(MLIRLinalgTransforms

  ${LIBS}
  MLIRLinalgTransformPatternsIncGen
  )
target_link_libraries(MLIRLinalgTransforms ${LIBS})
