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
  MLIRArithmeticDialect
  MLIRArmNeonDialect
  MLIRArmSVEDialect
  MLIRArmSVETransforms
  MLIRAMXDialect
  MLIRAMXTransforms
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRTargetLLVMIRExport
  MLIRTransforms
  MLIRVectorDialect
  MLIRVectorTransforms
  MLIRX86VectorDialect
  MLIRX86VectorTransforms
  )
