add_llvm_library(LLVMHexagonAsmPrinter
  HexagonInstPrinter.cpp
  )

add_dependencies(LLVMHexagonAsmPrinter HexagonCommonTableGen)
