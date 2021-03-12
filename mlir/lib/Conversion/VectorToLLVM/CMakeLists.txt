add_mlir_conversion_library(MLIRVectorToLLVM
  ConvertVectorToLLVM.cpp
  ConvertVectorToLLVMPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorToLLVM

  DEPENDS
  MLIRConversionPassIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArmNeon
  MLIRAVX512
  MLIRAVX512Transforms
  MLIRArmSVE
  MLIRArmSVEToLLVM
  MLIRLLVMArmSVE
  MLIRLLVMIR
  MLIRStandardToLLVM
  MLIRTargetLLVMIRExport
  MLIRTransforms
  MLIRVector
  )
