set(LLVM_OPTIONAL_SOURCES
  ConvertToLLVMPass.cpp
  ToLLVMInterface.cpp
)

add_mlir_conversion_library(MLIRConvertToLLVMInterface
  ToLLVMInterface.cpp

  DEPENDS
  MLIRConvertToLLVMInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)

add_mlir_conversion_library(MLIRConvertToLLVMPass
  ConvertToLLVMPass.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRConvertToLLVMInterface
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRPass
  MLIRRewrite
  MLIRSupport
  MLIRTransformUtils
  )
