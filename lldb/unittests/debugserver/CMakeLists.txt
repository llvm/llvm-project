# Note: debugserver is a Darwin-only implementation of a remote debugging
# server. It is not intended to be used on other platforms. The tests are here
# because using the LLDB Host API is convienent and allows testing of both parts
# of the debugserver communication path. If you are looking for a non-darwin
# remote debugging server, please use lldb-server.

include_directories(${LLDB_SOURCE_DIR}/tools/debugserver/source
                    ${LLDB_SOURCE_DIR}/tools/debugserver/source/MacOSX)

add_lldb_unittest(debugserverTests
  RNBSocketTest.cpp
  debugserver_LogCallback.cpp

  LINK_LIBS
    lldbDebugserverCommon
    lldbHost
  LINK_COMPONENTS
    Support
  )
