set(LLVM_NO_RTTI 1)

add_lldb_library(lldbUtility
  ARM_DWARF_Registers.cpp
  KQueue.cpp
  PseudoTerminal.cpp
  RefCounter.cpp
  SharingPtr.cpp
  StringExtractor.cpp
  StringExtractorGDBRemote.cpp
  TimeSpecTimeout.cpp
  )
