set(LLVM_OPTIONAL_SOURCES
  MlirReduceMain.cpp
)

set(LLVM_LINK_COMPONENTS
  Support
  )

add_mlir_library(MLIRReduceLib
  MlirReduceMain.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRReduce
  MLIRSupport
  )
