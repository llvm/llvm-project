add_mlir_conversion_library(MLIRAVX512ToLLVM
  ConvertAVX512ToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/AVX512ToLLVM

  DEPENDS
  MLIRConversionPassIncGen
)

target_link_libraries(MLIRAVX512ToLLVM
  PUBLIC
  MLIRAVX512
  MLIRLLVMAVX512
  MLIRLLVMIR
  MLIRStandardToLLVM
  MLIRTransforms
  LLVMCore
  LLVMSupport
  )
