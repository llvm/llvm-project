set(LLVM_LINK_COMPONENTS
  MC
  Support
)

add_llvm_unittest(RISCVMCMCTests
  RISCVMCInstAnalysis.cpp
)
