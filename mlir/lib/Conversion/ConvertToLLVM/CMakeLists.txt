set(LLVM_OPTIONAL_SOURCES
  ConvertToLLVMPass.cpp
  ToLLVMInterface.cpp
)

add_mlir_conversion_library(MLIRConvertToLLVMInterface
  ToLLVMInterface.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)

add_mlir_conversion_library(MLIRConvertToLLVMPass
  ConvertToLLVMPass.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRConvertToLLVMInterface
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRPass
  MLIRRewrite
  MLIRSupport
  MLIRTransformUtils
  )
