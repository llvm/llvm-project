add_mlir_library(MLIRPresburger
  IntegerPolyhedron.cpp
  Matrix.cpp
  Simplex.cpp

  DEPENDS
  MLIRBuiltinLocationAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
