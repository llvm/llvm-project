add_lldb_unittest(APITests
  SBCommandInterpreterTest.cpp
  SBLineEntryTest.cpp
  SBMutexTest.cpp

  SBAPITEST

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

# Apply -Wno-documentation-deprecated-sync while we migrate away from
# report_fatal_error in llvm/include/llvm/Support/ErrorHandling.h
# and llvm/include/llvm/Support/Error.h.
# Remove this block of code when the migration is complete.
# See https://github.com/llvm/llvm-project/issues/138914.
check_cxx_compiler_flag("-Wno-documentation-deprecated-sync"
                        CXX_SUPPORTS_NO_DOCUMENTATION_DEPRECATED_SYNC)
if (CXX_SUPPORTS_NO_DOCUMENTATION_DEPRECATED_SYNC)
  target_compile_options(APITests
    PRIVATE -Wno-documentation-deprecated-sync)
endif()

if(Python3_RPATH)
  set_property(TARGET APITests APPEND PROPERTY BUILD_RPATH "${Python3_RPATH}")
endif()
