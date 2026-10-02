set(LLVM_TARGET_DEFINITIONS InstCombineCLOptions.td)
tablegen(LLVM InstCombineCLOptions.inc -gen-opt-parser-defs)
add_public_tablegen_target(InstCombineCLOptionsTableGen)

add_llvm_component_library(LLVMInstCombine
  InstructionCombining.cpp
  InstCombineAddSub.cpp
  InstCombineAtomicRMW.cpp
  InstCombineAndOrXor.cpp
  InstCombineCalls.cpp
  InstCombineCasts.cpp
  InstCombineCLOptions.cpp
  InstCombineCompares.cpp
  InstCombineLoadStoreAlloca.cpp
  InstCombineMulDivRem.cpp
  InstCombineNegator.cpp
  InstCombinePHI.cpp
  InstCombineSelect.cpp
  InstCombineShifts.cpp
  InstCombineSimplifyDemanded.cpp
  InstCombineVectorOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/InstCombine

  DEPENDS
  InstCombineCLOptionsTableGen
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  Core
  Option
  Support
  TransformUtils
  )
