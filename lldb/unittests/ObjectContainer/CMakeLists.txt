add_lldb_unittest(ObjectContainerTests
  ObjectContainerUniversalMachOTest.cpp

  LINK_LIBS
    lldbPluginObjectContainerMachOArchive
    lldbPluginObjectContainerMachOFileset
    lldbCore
    lldbUtilityHelpers
    LLVMTestingSupport
  )
