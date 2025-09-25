add_lldb_unittest(EmulatorTests
  ARM64/TestAArch64Emulator.cpp
  LoongArch/TestLoongArchEmulator.cpp
  RISCV/TestRISCVEmulator.cpp
  
  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbPluginInstructionARM64
    lldbPluginInstructionLoongArch
    lldbPluginInstructionRISCV
  )

# Only add RISCV emulator tests if RISCV is in the list of LLVM targets to build.
# This is necessary because some buildbots (e.g., X86-only machines) 
# do not have RISCV support enabled. Without this check, the RISCV tests would be added
# causing build failures.
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
