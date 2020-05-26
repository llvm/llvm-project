set (SOURCES InlineFeaturesAnalysis.cpp)

add_llvm_library(LLVMMLPolicies STATIC
  ${SOURCES}

  DEPENDS
  intrinsics_gen

  )
