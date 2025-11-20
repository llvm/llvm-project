set(LLVM_LINK_COMPONENTS
  BinaryFormat
  DebugInfoDWARF
  DebugInfoDWARFLowLevel
  AllTargetsDescs
  AllTargetsInfos
  BinaryFormat
  MC
  Object
  Support
  TargetParser
  BinaryFormat
  )

add_llvm_tool(llvm-dwarfdump
  SectionSizes.cpp
  Statistics.cpp
  llvm-dwarfdump.cpp
  )

add_subdirectory(fuzzer)
