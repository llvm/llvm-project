add_library(lldb-mi-utils OBJECT
  ${LLDB_SOURCE_DIR}/tools/lldb-mi/MIUtilString.cpp
  )

add_lldb_unittest(LLDBMiUtilTests
  StringTest.cpp

  LINK_COMPONENTS
    Support
  )

target_sources(LLDBMiUtilTests PRIVATE $<TARGET_OBJECTS:lldb-mi-utils>)
