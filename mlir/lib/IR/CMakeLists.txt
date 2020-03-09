file(GLOB globbed *.c *.cpp)
add_mlir_library(MLIRIR
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRCallOpInterfacesIncGen
  MLIRSideEffectOpInterfacesIncGen
  MLIROpAsmInterfacesIncGen
  )
target_link_libraries(MLIRIR
  PUBLIC
  MLIRSupport
  LLVMSupport
  )
