add_mlir_conversion_library(MLIRLLVMCommonConversion
  ConversionTarget.cpp
  LoweringOptions.cpp
  MemRefBuilder.cpp
  Pattern.cpp
  PrintCallHelper.cpp
  StructBuilder.cpp
  TypeConverter.cpp
  VectorPattern.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRSupport
  MLIRTransforms
  )

set_target_properties(obj.MLIRLLVMCommonConversion PROPERTIES UNITY_BUILD_BATCH_SIZE 1)
