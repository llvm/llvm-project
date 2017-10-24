set(LLVM_LINK_COMPONENTS
  AllTargetsAsmPrinters
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  CFIVerify
  MC
  MCParser
  Object
  Support
  )

add_llvm_unittest(CFIVerifyTests
  FileAnalysis.cpp
  GraphBuilder.cpp)
target_link_libraries(CFIVerifyTests LLVMCFIVerify)
