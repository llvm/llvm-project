add_mlir_library(MLIRPresburger
  Barvinok.cpp
  IntegerRelation.cpp
  LinearTransform.cpp
  Matrix.cpp
  PresburgerRelation.cpp
  PresburgerSpace.cpp
  PWMAFunction.cpp
  QuasiPolynomial.cpp
  Simplex.cpp
  Utils.cpp

  LINK_LIBS PUBLIC
  MLIRSupport
  )
