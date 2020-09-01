# Note: debugserver is a Darwin-only implementation of a remote debugging
# server. It is not intended to be used on other platforms. The tests are here
# because using the LLDB Host API is convenient and allows testing of both parts
# of the debugserver communication path. If you are looking for a non-darwin
# remote debugging server, please use lldb-server.

add_lldb_unittest(debugserverTests
  JSONTest.cpp
  RNBSocketTest.cpp
  debugserver_LogCallback.cpp

  LINK_LIBS
    lldbDebugserverCommon
    lldbHost
    LLVMTestingSupport
  LINK_COMPONENTS
    Support
  )

target_include_directories(debugserverTests PRIVATE
  ${LLDB_SOURCE_DIR}/tools/debugserver/source
  ${LLDB_SOURCE_DIR}/tools/debugserver/source/MacOSX)

if(APPLE_EMBEDDED)
  set_property(TARGET debugserverTests APPEND PROPERTY COMPILE_DEFINITIONS
      WITH_LOCKDOWN
      WITH_FBS
      WITH_BKS
      )

  add_lldb_unittest(debugserverNonUITests
    JSONTest.cpp
    RNBSocketTest.cpp
    debugserver_LogCallback.cpp

    LINK_LIBS
      lldbDebugserverCommon_NonUI
      lldbHost
    LINK_COMPONENTS
      Support
    )
endif()
