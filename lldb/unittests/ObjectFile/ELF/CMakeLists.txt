add_lldb_unittest(ObjectFileELFTests
  TestELFHeader.cpp

  LINK_LIBS
    lldbPluginObjectFileELF
    lldbCore
  )
