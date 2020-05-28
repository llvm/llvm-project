set (SOURCES InlineFeaturesAnalysis.cpp)

add_llvm_library(LLVMMLPolicies
  ${SOURCES}

  DEPENDS
  intrinsics_gen

  LINK_LIBS
  LLVMCore
  )
