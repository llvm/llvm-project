add_lldb_unittest(APITests
  SBCommandInterpreterTest.cpp
  SBStructuredDataTest.cpp

  LINK_LIBS
    liblldb
  )

if(Python3_RPATH)
  set_property(TARGET APITests APPEND PROPERTY BUILD_RPATH "${Python3_RPATH}")
endif()
