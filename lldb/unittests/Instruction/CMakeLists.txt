if("ARM" IN_LIST LLVM_TARGETS_TO_BUILD)
  add_lldb_unittest(EmulatorTests
    TestAArch64Emulator.cpp
    LINK_LIBS
      lldbCore
      lldbSymbol
      lldbTarget
      lldbPluginInstructionARM64
    LINK_COMPONENTS
      Support
      ${LLVM_TARGETS_TO_BUILD})
endif()
