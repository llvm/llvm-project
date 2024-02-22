set(LLVM_LINK_COMPONENTS
  DebugInfoDWARF
  Support
  CAS
  MCCAS
  CASUtil
  BinaryFormat
  Object
  TargetParser
  )

add_llvm_tool(llvm-cas-dump
  llvm-cas-dump.cpp
  CASDWARFObject.cpp
  MCCASPrinter.cpp
  StatsCollector.cpp
)
