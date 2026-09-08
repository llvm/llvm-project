add_lldb_unittest(ObjectContainerTests
  ObjectContainerClangOffloadBundleTest.cpp
  ObjectContainerUniversalMachOTest.cpp

  LINK_LIBS
    lldbPluginObjectContainerClangOffloadBundle
    lldbPluginObjectContainerMachOArchive
    lldbPluginObjectContainerMachOFileset
    lldbPluginObjectFileELF
    lldbCore
    lldbUtilityHelpers
    LLVMTestingSupport
  )
