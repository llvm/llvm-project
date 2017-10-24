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

add_llvm_tool(llvm-cfi-verify
  llvm-cfi-verify.cpp)

add_subdirectory(lib)
