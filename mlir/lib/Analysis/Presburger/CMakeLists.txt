add_mlir_library(MLIRPresburger
  Barvinok.cpp
  IntegerRelation.cpp
  LinearTransform.cpp
  Matrix.cpp
  MPInt.cpp
  PresburgerRelation.cpp
  PresburgerSpace.cpp
  PWMAFunction.cpp
  QuasiPolynomial.cpp
  Simplex.cpp
  SlowMPInt.cpp
  Utils.cpp

  LINK_LIBS PUBLIC
  MLIRSupport
  )
