set(FILES "")
set(DEPS "")

if ("ARM" IN_LIST LLVM_TARGETS_TO_BUILD)
  list(APPEND FILES ARM64/TestAArch64Emulator.cpp)
  list(APPEND DEPS lldbPluginInstructionARM64)
endif ()

if ("RISCV" IN_LIST LLVM_TARGETS_TO_BUILD)
  list(APPEND FILES RISCV/TestRISCVEmulator.cpp)
  list(APPEND DEPS lldbPluginInstructionRISCV)
endif ()

list(LENGTH FILES LISTLEN)

if (LISTLEN GREATER 0)
  add_lldb_unittest(EmulatorTests
    ${FILES}

    LINK_LIBS
      lldbCore
      lldbSymbol
      lldbTarget
      ${DEPS}
    LINK_COMPONENTS
      Support
      ${LLVM_TARGETS_TO_BUILD}
    )
endif ()
