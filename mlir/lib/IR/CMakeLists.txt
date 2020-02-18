file(GLOB globbed *.c *.cpp)
add_llvm_library(MLIRIR
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  LINK_LIBS
  MLIRSupport
  LLVMSupport
  ${LLVM_PTHREAD_LIB}
  )
add_dependencies(MLIRIR MLIRCallOpInterfacesIncGen MLIROpAsmInterfacesIncGen MLIRSupport LLVMSupport)
