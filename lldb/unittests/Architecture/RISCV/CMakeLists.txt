if ("RISCV" IN_LIST LLVM_TARGETS_TO_BUILD)
  add_lldb_unittest(ArchitectureRISCVTests
    ArchitectureRISCVTest.cpp

    LINK_COMPONENTS
      Support
    LINK_LIBS
      lldbCore
      lldbPluginArchitectureRISCV
      lldbUtility)
endif()
