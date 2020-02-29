add_llvm_library(MLIRTestIR
  TestFunc.cpp
  TestMatchers.cpp
  TestSymbolUses.cpp

  ADDITIONAL_HEADER_DIRS

  LINK_LIBS
  MLIRIR
  MLIRPass
  MLIRStandardOps
  LLVMSupport
  )
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../TestDialect)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../TestDialect)
add_dependencies(MLIRTestIR
  MLIRTestDialect
  )
