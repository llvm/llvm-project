add_library(lldb-mi-utils OBJECT
  ${LLDB_SOURCE_DIR}/tools/lldb-mi/MIUtilString.cpp
  )

add_lldb_unittest(LLDBMiUtilTests
  $<TARGET_OBJECTS:lldb-mi-utils>
  StringTest.cpp

  LINK_COMPONENTS
    Support
  )

set_target_properties(lldb-mi-utils PROPERTIES FOLDER "lldb libraries")
