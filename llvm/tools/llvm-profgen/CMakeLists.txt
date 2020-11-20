set(LLVM_LINK_COMPONENTS
  Core
  ProfileData
  Support
  Symbolize
  )

add_llvm_tool(llvm-profgen
  llvm-profgen.cpp
  PerfReader.cpp
  )
