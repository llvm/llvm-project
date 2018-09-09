# FIXME: Test is failing Swift PR testing: rdar://problem/44270082
#if("ARM" IN_LIST LLVM_TARGETS_TO_BUILD)
#  add_lldb_unittest(DisassemblerTests
#    TestArmv7Disassembly.cpp
#    LINK_LIBS
#      lldbCore
#      lldbSymbol
#      lldbTarget
#      lldbPluginDisassemblerLLVM
#      lldbPluginProcessUtility
#    LINK_COMPONENTS
#      Support
#      ${LLVM_TARGETS_TO_BUILD})
#endif()
