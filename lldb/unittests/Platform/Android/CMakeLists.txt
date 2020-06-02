include_directories(${LLDB_SOURCE_DIR}/source/Plugins/Platform/Android)

add_lldb_unittest(AdbClientTest
  AdbClientTest.cpp

  LINK_LIBS
    lldbPluginPlatformAndroid
  )
