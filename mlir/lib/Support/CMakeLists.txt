set(MLIRSupport_Sources
  FileUtilities.cpp
  InterfaceSupport.cpp
  RawOstreamExtras.cpp
  StateStack.cpp
  StorageUniquer.cpp
  Timing.cpp
  ToolUtilities.cpp
  TypeID.cpp
)

add_mlir_library(MLIRSupport
  ${MLIRSupport_Sources}
  ADDITIONAL_HEADER_DIRS ${MLIR_MAIN_INCLUDE_DIR}/mlir/Support
  LINK_LIBS PUBLIC ${LLVM_PTHREAD_LIB}
)

# Create a tablegen "version" of MLIRSupport to be used by MLIR TableGen tools.
# It does not link to libLLVM.so so it can be used in tools statically linking
# libLLVMSupport.
# This is necessary to prevent ODR violations when linking MLIR TableGen tools
# with MLIRSupport.
if(LLVM_LINK_LLVM_DYLIB)
  add_mlir_library(MLIRSupport_tablegen
    ${MLIRSupport_Sources}
    EXCLUDE_FROM_LIBMLIR
    DISABLE_INSTALL
    DISABLE_LLVM_LINK_LLVM_DYLIB

    LINK_LIBS PUBLIC ${LLVM_PTHREAD_LIB}
  )
else()
  add_library(MLIRSupport_tablegen ALIAS MLIRSupport)
endif()
