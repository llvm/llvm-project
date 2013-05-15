set(LLVM_NO_RTTI 1)

include_directories(${LIBXML2_INCLUDE_DIR})

add_lldb_library(lldbPluginSymbolVendorMacOSX
  SymbolVendorMacOSX.cpp
  )
