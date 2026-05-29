include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)

add_llvm_component_library(LLVMMetalAIRWriter
  BitcodeEmitter.cpp
  ConstantsWriter.cpp
  FunctionWriter.cpp
  MetadataWriter.cpp
  MetalEmbedderPass.cpp
  MetallibWriter.cpp
  MetalWriterPass.cpp
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
  Metal
  )
