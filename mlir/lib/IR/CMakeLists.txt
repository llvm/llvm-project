file(GLOB globbed *.c *.cpp)
add_llvm_library(MLIRIR
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRCallOpInterfacesIncGen
  MLIROpAsmInterfacesIncGen

  LINK_LIBS
  MLIRSupport
  LLVMSupport
  ${LLVM_PTHREAD_LIB}
  )
