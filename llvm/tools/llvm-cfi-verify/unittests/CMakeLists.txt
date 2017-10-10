set(LLVM_LINK_COMPONENTS
  AllTargetsAsmPrinters
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  MC
  Object
  MCParser
  Object
  Support
  )



add_llvm_unittest(CFIVerifyTests
  FileAnalysis.cpp ../FileAnalysis.cpp)
