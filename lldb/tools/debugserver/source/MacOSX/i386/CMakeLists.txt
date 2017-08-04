include_directories(${LLDB_SOURCE_DIR}/tools/debugserver/source)
add_library(lldbDebugserverMacOSX_I386
  DNBArchImplI386.cpp
  )
