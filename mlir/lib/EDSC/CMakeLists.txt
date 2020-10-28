set(LLVM_OPTIONAL_SOURCES
  Builders.cpp
  CoreAPIs.cpp
  )

add_mlir_library(MLIREDSC
  Builders.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/EDSC

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  )

