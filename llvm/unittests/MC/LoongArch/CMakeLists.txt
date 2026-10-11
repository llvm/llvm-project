set(LLVM_LINK_COMPONENTS
  LoongArchDesc
  LoongArchDisassembler
  LoongArchInfo
  MC
  MCDisassembler
  Support
  TargetParser
  )

add_llvm_unittest(LoongArchMCTests
  LoongArchMCDisassemblerTest.cpp
  )
