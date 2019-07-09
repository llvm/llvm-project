# Due to sources including headers like:
# #include "MacOSX/i386/DNBArchImplI386.h"
# we must include the grandparent directory...
include_directories(${LLDB_SOURCE_DIR}/tools/debugserver/source)

add_library(lldbDebugserverDarwin_DarwinLog
  ActivityStore.cpp
  DarwinLogCollector.cpp
  LogFilter.cpp
  LogFilterChain.cpp
  LogFilterExactMatch.cpp
  LogFilterRegex.cpp
  LogMessage.cpp
  LogMessageOsLog.cpp
  )

set_target_properties(lldbDebugserverDarwin_DarwinLog PROPERTIES FOLDER "lldb libraries/debugserver")
