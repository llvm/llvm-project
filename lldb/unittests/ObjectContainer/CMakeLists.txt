add_lldb_unittest(ObjectContainerTests
  ObjectContainerUniversalMachOTest.cpp

  LINK_LIBS
    lldbPluginObjectContainerMachOArchive
    lldbPluginObjectContainerMachOFileset
    lldbPluginObjectFileMachO
    lldbCore
    lldbUtilityHelpers
    LLVMTestingSupport
  )
