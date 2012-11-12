add_llvm_library(LLVMHexagonDesc
  HexagonMCTargetDesc.cpp
  HexagonMCAsmInfo.cpp
  )

add_dependencies(LLVMHexagonDesc HexagonCommonTableGen)
