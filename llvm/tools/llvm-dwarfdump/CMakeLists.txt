set(LLVM_LINK_COMPONENTS
  DebugInfoDWARF
  DebugInfoDWARFLowLevel
  AllTargetsDescs
  AllTargetsInfos
  MC
  Object
  Support
  TargetParser
  )

add_llvm_tool(llvm-dwarfdump
  SectionSizes.cpp
  Statistics.cpp
  llvm-dwarfdump.cpp
  )

add_subdirectory(fuzzer)
