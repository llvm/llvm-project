set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  MC
  MCParser
  Support
  TargetParser
  )

add_llvm_tool(llvm-mc
  llvm-mc.cpp
  Disassembler.cpp
  )

target_link_libraries(llvm-mc PRIVATE LLVMBOLTCore LLVMBOLTTargetX86)
target_include_directories(llvm-mc PRIVATE "${LLVM_MAIN_SRC_DIR}/../bolt/include")
