add_lldb_unittest(LLDBMCPTests
  MultiplexerTest.cpp
  ${LLDB_SOURCE_DIR}/tools/lldb-mcp/Multiplexer.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbHost
    lldbProtocolMCP
    lldbUtility
    LLVMTestingSupport
  )

target_include_directories(LLDBMCPTests PRIVATE
  ${LLDB_SOURCE_DIR}/tools/lldb-mcp)
