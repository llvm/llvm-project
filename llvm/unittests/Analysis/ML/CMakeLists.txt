set(LLVM_LINK_COMPONENTS
  Analysis  
  AsmParser
  Core
  Support
  TransformUtils
  )

add_llvm_unittest(MLAnalysisTests
  InlineFeaturesAnalysisTest.cpp
  )
