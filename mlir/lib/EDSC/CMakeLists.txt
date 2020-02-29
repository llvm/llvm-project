set(LLVM_OPTIONAL_SOURCES
  Builders.cpp
  CoreAPIs.cpp
  )

add_llvm_library(MLIREDSC
  Builders.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/EDSC

  LINK_LIBS
  MLIRIR
  MLIRSupport
  LLVMSupport
  )

add_llvm_library(MLIREDSCInterface
  CoreAPIs.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/EDSC

  LINK_LIBS
  MLIRIR
  MLIRSupport
  MLIRParser
  LLVMSupport
  )
