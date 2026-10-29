add_lldb_library(lldbPluginProcessAIX
  NativeProcessAIX.cpp
  PthdbWrapper.cpp
  NativeRegisterContextAIX.cpp
  NativeRegisterContextAIX_ppc64.cpp
  NativeThreadAIX.cpp
  NativeRegisterContextAIX.cpp
  NativeRegisterContextAIX_ppc64.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginProcessPOSIX
    lldbPluginProcessUtility
  )

target_compile_definitions(lldbPluginProcessAIX PRIVATE "-D_ALL_SOURCE")

# On AIX, link against libpthdebug for pthreads debugging support.
if(CMAKE_SYSTEM_NAME STREQUAL "AIX")
  target_link_libraries(lldbPluginProcessAIX PRIVATE -lpthdebug)
endif()
