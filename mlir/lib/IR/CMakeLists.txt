# TODO: Remove the need for global constructors within IR/
add_flag_if_supported("-Wno-global-constructors" WNO_GLOBAL_CONSTRUCTOR_MLIR_IR)

file(GLOB globbed *.c *.cpp)
add_mlir_library(MLIRIR
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRCallInterfacesIncGen
  MLIROpAsmInterfacesIncGen
  )
target_link_libraries(MLIRIR
  PUBLIC
  MLIRSupport
  LLVMSupport
  )
