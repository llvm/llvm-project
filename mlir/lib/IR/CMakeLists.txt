file(GLOB globbed *.c *.cpp)
add_llvm_library(MLIRIR
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRCallOpInterfacesIncGen
  MLIROpAsmInterfacesIncGen
  )
target_link_libraries(MLIRIR
  PUBLIC
  MLIRSupport
  LLVMSupport
  )
