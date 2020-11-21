
set(LLVM_LINK_COMPONENTS
  AllTargetsDescs
  AllTargetsDisassemblers
  Core
  MC
  MCDisassembler
  Object
  Support
  Symbolize
  )

add_llvm_tool(llvm-profgen
  llvm-profgen.cpp
  PerfReader.cpp
  ProfiledBinary.cpp
  )
