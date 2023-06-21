set(LLVM_OPTIONAL_SOURCES
  FileUtilities.cpp
  InterfaceSupport.cpp
  StorageUniquer.cpp
  Timing.cpp
  ToolUtilities.cpp
  TypeID.cpp
)

add_mlir_library(MLIRSupport
  FileUtilities.cpp
  InterfaceSupport.cpp
  StorageUniquer.cpp
  Timing.cpp
  ToolUtilities.cpp
  TypeID.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Support

  LINK_LIBS PUBLIC
  ${LLVM_PTHREAD_LIB})
