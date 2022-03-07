include_directories(
  ${CMAKE_SOURCE_DIR}/lib/Target/X86
  ${CMAKE_BINARY_DIR}/lib/Target/X86
  )

set(LLVM_LINK_COMPONENTS
  MC
  MCDisassembler
  Target
  X86Desc
  X86Disassembler
  )

add_llvm_unittest(X86MCTests
  X86MCDisassemblerTest.cpp
  )
