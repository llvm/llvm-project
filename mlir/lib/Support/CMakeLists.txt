set(LLVM_OPTIONAL_SOURCES
  FileUtilities.cpp
  MlirOptMain.cpp
  StorageUniquer.cpp
  ToolUtilities.cpp
)

add_mlir_library(MLIRSupport
  FileUtilities.cpp
  StorageUniquer.cpp
  ToolUtilities.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Support

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  ${LLVM_PTHREAD_LIB})

add_mlir_library(MLIROptLib
  MlirOptMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Support

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRParser
  MLIRSupport
  )
