set(LLVM_LINK_COMPONENTS
  DebugInfoDWARF
  AllTargetsDescs
  AllTargetsInfos
  MC
  Object
  Support
  )

add_llvm_tool(llvm-dwarfdump
  llvm-dwarfdump.cpp
  )

add_subdirectory(fuzzer)
