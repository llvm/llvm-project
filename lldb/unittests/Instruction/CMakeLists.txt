add_lldb_unittest(EmulatorTests
  ARM64/TestAArch64Emulator.cpp
  LoongArch/TestLoongArchEmulator.cpp
  RISCV/TestRISCVEmulator.cpp
  
  LINK_COMPONENTS
    Support
    ${LLVM_TARGETS_TO_BUILD}
  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbPluginInstructionARM64
    lldbPluginInstructionLoongArch
    lldbPluginInstructionRISCV
    lldbPluginProcessUtility
  )

if ("RISCV" IN_LIST LLVM_TARGETS_TO_BUILD)
  add_lldb_unittest(RISCVEmulatorTests
    RISCV/TestRiscvInstEmulation.cpp

    LINK_COMPONENTS
      Support
      ${LLVM_TARGETS_TO_BUILD}
    LINK_LIBS
      lldbCore
      lldbSymbol
      lldbTarget
      lldbPluginInstructionRISCV
      lldbPluginDisassemblerLLVMC
      lldbPluginUnwindAssemblyInstEmulation
      lldbPluginProcessUtility
    )
endif()
