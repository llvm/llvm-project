set(LLVM_LINK_COMPONENTS
  MC
  MCDisassembler
  Target
  X86Desc
  X86Disassembler
  X86Info
  )

add_llvm_unittest(X86MCTests
  X86MCDisassemblerTest.cpp
  )
