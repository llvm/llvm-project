include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)

add_llvm_component_library(LLVMAIRWriter
  BitcodeEmitter.cpp
  ConstantsWriter.cpp
  FunctionWriter.cpp
  LowerVectorSelect.cpp
  MetadataWriter.cpp
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
