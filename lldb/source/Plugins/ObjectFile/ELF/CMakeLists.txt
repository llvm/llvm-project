set(LLVM_NO_RTTI 1)

add_lldb_library(lldbPluginObjectFileELF
  ELFHeader.cpp
  ObjectFileELF.cpp
  )
