add_lldb_unittest(APITests
  SBCommandInterpreterTest.cpp

  LINK_LIBS
    liblldb
  )

if(PYTHON_RPATH)
  set_property(TARGET APITests APPEND PROPERTY BUILD_RPATH "${PYTHON_RPATH}")
endif()
