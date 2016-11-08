if ("AArch64" IN_LIST LLVM_TARGETS_TO_BUILD)
  add_lldb_unittest(InstEmulationTests TestArm64InstEmulation.cpp)
endif()
