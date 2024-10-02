add_mlir_conversion_library(MLIRVectorToLLVM
  PARTIAL_SOURCES_INTENDED
  ConvertVectorToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorToLLVM

  DEPENDS
  MLIRConversionPassIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithAttrToLLVMConversion
  MLIRArithDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRTargetLLVMIRExport
  MLIRTransforms
  MLIRVectorDialect
  MLIRVectorTransforms
  )

add_mlir_conversion_library(MLIRVectorToLLVMPass
  PARTIAL_SOURCES_INTENDED

  ConvertVectorToLLVMPass.cpp
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorToLLVM

  LINK_LIBS PUBLIC
  MLIRVectorToLLVM

  MLIRArmNeonDialect
  MLIRArmSMEDialect
  MLIRArmSVEDialect
  MLIRArmSVETransforms
  MLIRAMXDialect
  MLIRAMXTransforms
  MLIRX86VectorDialect
  MLIRX86VectorTransforms
)
