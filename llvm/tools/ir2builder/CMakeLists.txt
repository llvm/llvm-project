set(LLVM_LINK_COMPONENTS
  Support
  AsmParser
  Core
  )

add_llvm_utility(ir2builder
  ir2builder.cpp
  )