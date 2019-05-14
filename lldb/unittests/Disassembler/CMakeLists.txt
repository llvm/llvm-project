if("ARM" IN_LIST LLVM_TARGETS_TO_BUILD)
  add_lldb_unittest(DisassemblerTests
    TestArm64Disassembly.cpp
    TestArmv7Disassembly.cpp
    LINK_LIBS
      lldbCore
      lldbSymbol
      lldbTarget
      lldbPluginDisassemblerLLVM
      lldbPluginProcessUtility
    LINK_COMPONENTS
      Support
      ${LLVM_TARGETS_TO_BUILD})
endif()
