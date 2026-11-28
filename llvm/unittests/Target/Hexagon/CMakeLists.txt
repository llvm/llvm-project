set(LLVM_LINK_COMPONENTS
  HexagonCodeGen
  HexagonDesc
  HexagonInfo
  Support
  Target
  )

add_llvm_target_unittest(HexagonTests
  HexagonRelocationTest.cpp
  )
