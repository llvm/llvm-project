set(disas_srcs "")

if("ARM" IN_LIST LLVM_TARGETS_TO_BUILD)
  list(APPEND
    ARM/TestArm64Disassembly.cpp
    ARM/TestArmv7Disassembly.cpp
  )
endif()

if("X86" IN_LIST LLVM_TARGETS_TO_BUILD)
  list(APPEND disas_srcs
    x86/TestGetControlFlowKindx86.cpp
  )
endif()

if("RISCV" IN_LIST LLVM_TARGETS_TO_BUILD)
  list(APPEND disas_srcs
    RISCV/TestMCDisasmInstanceRISCV.cpp
  )
endif()

if (disas_srcs)
  add_lldb_unittest(DisassemblerTests
    ${disas_srcs}
    LINK_COMPONENTS
      Support
      ${LLVM_TARGETS_TO_BUILD}
    LINK_LIBS
      lldbCore
      lldbSymbol
      lldbTarget
      lldbPluginDisassemblerLLVMC
      lldbPluginProcessUtility
    )
endif()
