set(PROTOCOL_TEST_SOURCES
  ProtocolMCPTest.cpp
)

if(NOT WIN32)
  list(APPEND PROTOCOL_TEST_SOURCES ProtocolMCPServerTest.cpp)
endif()

add_lldb_unittest(ProtocolTests
  ${PROTOCOL_TEST_SOURCES}

  LINK_LIBS
    lldbHost
    lldbProtocolMCP
    lldbUtility
    LLVMTestingSupport
)
