set(LLVM_LINK_COMPONENTS
  Core
  OrcJIT
  OrcShared
  Support
  nativecodegen
  )

add_llvm_example(HowToUseLLJIT
  HowToUseLLJIT.cpp

  EXPORT_SYMBOLS
  )
