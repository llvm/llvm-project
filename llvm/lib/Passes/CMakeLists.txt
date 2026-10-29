set(LLVM_TABLEGEN_PROJECT LLVM_HEADERS)
set(LLVM_TARGET_DEFINITIONS PassesOptions.td)
tablegen(LLVM PassesOptions.inc -gen-opt-parser-defs)
add_public_tablegen_target(PassesOptionsTableGen)

add_llvm_component_library(LLVMPasses
  CodeGenPassBuilder.cpp
  PassBuilder.cpp
  PassBuilderBindings.cpp
  PassBuilderPipelines.cpp
  PassesOptions.cpp
  RunCodeGen.cpp
  StandardInstrumentations.cpp
  TriggerCrashPasses.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Passes

  DEPENDS
  PassesOptionsTableGen
  intrinsics_gen

  LINK_COMPONENTS
  AggressiveInstCombine
  Analysis
  AsmPrinter
  CFGuard
  CodeGen
  GlobalISel
  Core
  Coroutines
  HipStdPar
  IPO
  InstCombine
  IRPrinter
  ObjCARC
  Option
  Scalar
  Support
  Target
  TransformUtils
  Vectorize
  Instrumentation
  )
