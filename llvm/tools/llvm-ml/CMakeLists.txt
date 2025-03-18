set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  MC
  MCParser
  Option
  Support
  TargetParser
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)

tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(MLTableGen)

add_llvm_tool(llvm-ml
  llvm-ml.cpp
  Disassembler.cpp
  DEPENDS
  MLTableGen
  GENERATE_DRIVER
  )

llvm_add_tool_symlink(LLD llvm-ml64 llvm-ml ALWAYS_GENERATE)
llvm_install_symlink(LLD llvm-ml64 llvm-ml ALWAYS_GENERATE)
