set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  MC
  MCParser
  Object
  Support
  Symbolize
  TargetParser
  )

add_llvm_unittest(CFIVerifyTests
  FileAnalysis.cpp
  GraphBuilder.cpp
  )
target_link_libraries(CFIVerifyTests PRIVATE LLVMCFIVerify)
