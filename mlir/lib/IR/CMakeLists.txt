file(GLOB globbed *.c *.cpp)
add_mlir_library(MLIRIR
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRCallInterfacesIncGen
  MLIROpAsmInterfacesIncGen
  MLIRSymbolInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRSupport
  LLVMSupport
  )
