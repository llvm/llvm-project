include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)

add_llvm_component_library(LLVMAIRWriter
  AggregateScalarize.cpp
  BitcodeEmitter.cpp
  ConstantExprLower.cpp
  ConstantsWriter.cpp
  CoopTensorLowering.cpp
  FunctionWriter.cpp
  IntegerLegalize.cpp
  LowerPointerVectors.cpp
  LowerVectorSelect.cpp
  MetadataWriter.cpp
  NormalizeGEPs.cpp
  PointeeRules.cpp
  PointerPointeeRepair.cpp
  AIREmbedderPass.cpp
  AIRLibWriter.cpp
  AIRWriterPass.cpp
  PointeeTypeMap.cpp
  TypeTableWriter.cpp
  ValueEnumerator.cpp

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  BitWriter
  Core
  MC
  Support
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  AIR
  )
