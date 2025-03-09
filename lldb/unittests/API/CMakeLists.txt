add_lldb_unittest(APITests
  SBCommandInterpreterTest.cpp

  LINK_LIBS
    liblldb
  )

# Build with -Wdocumentation. This relies on the tests including all the API
# headers through API/LLDB.h.
check_cxx_compiler_flag("-Wdocumentation"
                        CXX_SUPPORTS_DOCUMENTATION)
if (CXX_SUPPORTS_DOCUMENTATION)
  target_compile_options(APITests
    PRIVATE -Wdocumentation)
endif()


if(Python3_RPATH)
  set_property(TARGET APITests APPEND PROPERTY BUILD_RPATH "${Python3_RPATH}")
endif()
