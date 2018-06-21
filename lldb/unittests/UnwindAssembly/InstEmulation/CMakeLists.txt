if ("AArch64" IN_LIST LLVM_TARGETS_TO_BUILD)
  add_lldb_unittest(InstEmulationTests
    TestArm64InstEmulation.cpp
    LINK_LIBS
      lldbCore
      lldbSymbol
      lldbTarget
      lldbPluginUnwindAssemblyInstEmulation
      lldbPluginDisassemblerLLVM
      lldbPluginInstructionARM64
      lldbPluginProcessUtility
    LINK_COMPONENTS
      Support
      ${LLVM_TARGETS_TO_BUILD})
endif()
