add_llvm_library(LLVMInstCombine
  InstructionCombining.cpp
  InstCombineAddSub.cpp
  InstCombineAndOrXor.cpp
  InstCombineCalls.cpp
  InstCombineCasts.cpp
  InstCombineCompares.cpp
  InstCombineLoadStoreAlloca.cpp
  InstCombineMulDivRem.cpp
  InstCombinePHI.cpp
  InstCombineSelect.cpp
  InstCombineShifts.cpp 
  InstCombineSimplifyDemanded.cpp
  InstCombineVectorOps.cpp
  )

add_dependencies(LLVMInstCombine intrinsics_gen)
