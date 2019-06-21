# Create object library to avoid unnecessary linking. If CMake version
# doesn't support it, just create an ordinary library.
if(NOT (${CMAKE_VERSION} VERSION_LESS "3.5.0"))
  set (LIBRARY_TYPE "OBJECT")
endif()

add_library(lldb-mi-utils ${LIBRARY_TYPE}
  ${LLDB_SOURCE_DIR}/tools/lldb-mi/MIUtilString.cpp
  )

add_lldb_unittest(LLDBMiUtilTests
  StringTest.cpp

  LINK_COMPONENTS
    Support
  )

if((${CMAKE_VERSION} VERSION_LESS "3.5.0") OR
    NOT (${CMAKE_VERSION} VERSION_LESS "3.12.0"))
  # Link to either usual (cmake version < 3.5) or
  # object (cmake version >= 3.12) library with lldm-mi sources.
  target_link_libraries(LLDBMiUtilTests PRIVATE lldb-mi-utils)
else()
  # Object libraries still cannot be on the right-hand side of
  # `target_link_libraries` but `$<TARGET_OBJECTS:objlib>` expression is
  # already supported in `target_sources`.
  target_sources(LLDBMiUtilTests PRIVATE $<TARGET_OBJECTS:lldb-mi-utils>)
endif()

set_target_properties(lldb-mi-utils PROPERTIES FOLDER "lldb libraries")
