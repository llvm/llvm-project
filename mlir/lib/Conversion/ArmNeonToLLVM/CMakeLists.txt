add_mlir_conversion_library(MLIRArmNeonToLLVM
  ArmNeonToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArmNeonToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArmNeon
  MLIRLLVMArmNeon
  MLIRLLVMIR
  MLIRStandardToLLVM
  MLIRTransforms
  )
