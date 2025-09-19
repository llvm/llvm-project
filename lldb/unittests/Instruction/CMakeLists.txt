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
