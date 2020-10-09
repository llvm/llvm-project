set(LLVM_OPTIONAL_SOURCES
  FileUtilities.cpp
  IndentedOstream.cpp
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

  LINK_LIBS PUBLIC
  ${LLVM_PTHREAD_LIB})

add_mlir_library(MLIROptLib
  MlirOptMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Support

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRParser
  MLIRSupport
  )

# This doesn't use add_mlir_library as it is used in mlir-tblgen and else
# mlir-tblgen ends up depending on mlir-generic-headers.
add_llvm_library(MLIRSupportIndentedOstream
  IndentedOstream.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB

  LINK_COMPONENTS
  Support
  )
