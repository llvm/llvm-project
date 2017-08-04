# Due to sources including headers like:
# #include "MacOSX/i386/DNBArchImplI386.h"
# we must include the grandparent directory...
include_directories(${LLDB_SOURCE_DIR}/tools/debugserver/source)

add_library(lldbDebugserverMacOSX_X86_64
  DNBArchImplX86_64.cpp
  )
