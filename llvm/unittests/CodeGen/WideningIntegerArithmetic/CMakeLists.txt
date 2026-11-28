set(LLVM_LINK_COMPONENTS Core Support)

set(WIA_ANALYSIS_TEST_SOURCES)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  set(WIA_ANALYSIS_TEST_SOURCES WideningIntegerArithmeticAnalysisTest.cpp)
  if(MSVC)
    set_source_files_properties(WideningIntegerArithmeticAnalysisTest.cpp
      PROPERTIES COMPILE_OPTIONS /clang:-fno-access-control)
  else()
    set_source_files_properties(WideningIntegerArithmeticAnalysisTest.cpp
      PROPERTIES COMPILE_OPTIONS -fno-access-control)
  endif()
endif()

add_llvm_unittest(WideningIntegerArithmeticTests
  WideningIntegerArithmeticTest.cpp
  ${WIA_ANALYSIS_TEST_SOURCES}
  )
