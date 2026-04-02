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

# 'using namespace mlir' causes Type and other core type ambiguity.
set_source_files_properties(
  StructBuilder.cpp
  TypeConverter.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
