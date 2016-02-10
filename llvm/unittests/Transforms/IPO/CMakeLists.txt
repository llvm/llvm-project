set(LLVM_LINK_COMPONENTS
  Core
  Support
  IPO
  )

add_llvm_unittest(IPOTests
  LowerBitSets.cpp
  WholeProgramDevirt.cpp
  )
