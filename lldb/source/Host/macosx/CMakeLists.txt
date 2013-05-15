set(LLVM_NO_RTTI 1)

include_directories(${LIBXML2_INCLUDE_DIR})

add_lldb_library(lldbHostMacOSX
  Host.mm
  Symbols.cpp
  cfcpp/CFCBundle.cpp
  cfcpp/CFCData.cpp
  cfcpp/CFCMutableArray.cpp
  cfcpp/CFCMutableDictionary.cpp
  cfcpp/CFCMutableSet.cpp
  cfcpp/CFCString.cpp
  )
