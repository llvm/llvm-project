add_lldb_unittest(ObjectContainerTests
  ObjectContainerUniversalMachOTest.cpp

  LINK_LIBS
    lldbPluginObjectContainerMachOArchive
    lldbCore
    lldbUtilityHelpers
    LLVMTestingSupport
  )
