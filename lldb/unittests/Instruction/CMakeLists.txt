add_lldb_unittest(EmulatorTests
  ARM64/TestAArch64Emulator.cpp
  RISCV/TestRISCVEmulator.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbPluginInstructionARM64
    lldbPluginInstructionRISCV

  LINK_COMPONENTS
    Support
  )