set(LLVM_NO_RTTI 1)

add_lldb_library(lldbHostCommon
  Condition.cpp
  DynamicLibrary.cpp
  File.cpp
  FileSpec.cpp
  Host.cpp
  Mutex.cpp
  SocketAddress.cpp
  Symbols.cpp
  Terminal.cpp
  TimeValue.cpp
  )
