add_mlir_conversion_library(MLIRLLVMCommonConversion
  LoweringOptions.cpp
  MemRefBuilder.cpp
  StructBuilder.cpp
  TypeConverter.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMIR
  MLIRSupport
  MLIRTransforms
  )
