# Keep the complete public library as an umbrella over its components.
add_mlir_library(MLIRPresburgerIntegerRelations
  OBJECT
  IntegerRelation.cpp
  LinearTransform.cpp
  Matrix.cpp
  PresburgerRelation.cpp
  PresburgerSpace.cpp
  PWMAFunction.cpp
  Simplex.cpp
  Utils.cpp

  PARTIAL_SOURCES_INTENDED

  )

add_mlir_library(MLIRPresburger
  Barvinok.cpp
  QuasiPolynomial.cpp

  LINK_LIBS PUBLIC
  MLIRPresburgerIntegerRelations
  )

# Preserve the complete public archive. The component objects are compiled
# once; libMLIR still collects each object's defining library exactly once.
if(NOT BUILD_SHARED_LIBS)
  target_sources(MLIRPresburger PRIVATE $<TARGET_OBJECTS:obj.MLIRPresburgerIntegerRelations>)
endif()
