
set(LLVM_LINK_COMPONENTS
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  Core
  MC
  MCDisassembler
  Object
  ProfileData
  Support
  Symbolize
  )

add_llvm_tool(llvm-profgen
  llvm-profgen.cpp
  PerfReader.cpp
  ProfiledBinary.cpp
  ProfileGenerator.cpp
  )
