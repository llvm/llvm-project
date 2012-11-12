include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMIntelJITEvents
  IntelJITEventListener.cpp
  jitprofiling.c
  )
