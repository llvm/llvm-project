add_llvm_library(LLVMSystem
  Alarm.cpp
  Disassembler.cpp
  DynamicLibrary.cpp
  Host.cpp
  IncludeFile.cpp
  Memory.cpp
  Mutex.cpp
  Path.cpp
  Process.cpp
  Program.cpp
  Signals.cpp
  TimeValue.cpp
  )

if( BUILD_SHARED_LIBS AND NOT WIN32 )
  target_link_libraries(LLVMSystem dl)
endif()
