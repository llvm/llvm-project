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

list(FIND LLVM_ARGETS_TO_BUILD "X86" x86_idx)
if (NOT x86_idx LESS 0)
  add_llvm_unittest(CFIVerifyTests
    FileAnalysis.cpp)
endif()
