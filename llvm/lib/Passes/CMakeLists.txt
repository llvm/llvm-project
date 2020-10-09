add_llvm_component_library(LLVMPasses
  PassBuilder.cpp
  PassPlugin.cpp
  StandardInstrumentations.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Passes

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  AggressiveInstCombine
  Analysis
  Core
  Coroutines
  HelloNew
  IPO
  InstCombine
  ObjCARC
  Scalar
  Support
  Target
  TransformUtils
  Vectorize
  Instrumentation
  )
