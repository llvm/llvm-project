set(LLVM_OPTIONAL_SOURCES
  MaskingInterfaces.cpp
  )

function(add_mlir_interface_library name)
  add_mlir_library(MLIR${name}
    ${name}.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Vector/Interfaces

    DEPENDS
    MLIR${name}IncGen

    LINK_LIBS PUBLIC
    MLIRIR
    )
endfunction(add_mlir_interface_library)

add_mlir_interface_library(MaskingInterfaces)

