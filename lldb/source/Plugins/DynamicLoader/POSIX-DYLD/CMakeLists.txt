set(LLVM_NO_RTTI 1)

add_lldb_library(lldbPluginDynamicLoaderPosixDYLD
  AuxVector.cpp
  DYLDRendezvous.cpp
  DynamicLoaderPOSIXDYLD.cpp
  )
