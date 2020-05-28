set(LLVM_LINK_COMPONENTS
  Analysis  
  AsmParser
  Core
  MLPolicies
  Support
  TransformUtils
  )

add_llvm_unittest(MLAnalysisTests
  InlineFeaturesAnalysisTest.cpp
  )
