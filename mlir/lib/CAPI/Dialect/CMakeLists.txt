# TODO: Make the check source feature optional as an argument on *_add_library.
set(LLVM_OPTIONAL_SOURCES
  Linalg.cpp
  LinalgPasses.cpp
  SCF.cpp
  Shape.cpp
  Standard.cpp
  Tensor.cpp
)

add_mlir_public_c_api_library(MLIRCAPILinalg
  Linalg.cpp
  LinalgPasses.cpp

  DEPENDS
  MLIRLinalgPassIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRLinalg
  MLIRPass
  MLIRLinalgTransforms
)

add_mlir_public_c_api_library(MLIRCAPISCF
  SCF.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRSCF
)

add_mlir_public_c_api_library(MLIRCAPIShape
  Shape.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRShape
)

add_mlir_public_c_api_library(MLIRCAPIStandard
  Standard.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRStandard
)

add_mlir_public_c_api_library(MLIRCAPITensor
  Tensor.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRTensor
)
