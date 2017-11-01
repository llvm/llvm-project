set(LLVM_LINK_COMPONENTS
  AllTargetsAsmPrinters
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  DebugInfoDWARF
  MC
  MCParser
  Object
  Support
  )

add_llvm_tool(llvm-cfi-verify
  llvm-cfi-verify.cpp)

add_subdirectory(lib)
target_link_libraries(llvm-cfi-verify LLVMCFIVerify)
