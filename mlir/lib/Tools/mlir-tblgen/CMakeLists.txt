llvm_add_library(MLIRTblgenLib STATIC
  MlirTblgenMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/mlir-tblgen

  DISABLE_LLVM_LINK_LLVM_DYLIB

  LINK_COMPONENTS
  TableGen

  LINK_LIBS PUBLIC
  MLIRTableGen
  )

mlir_check_all_link_libraries(MLIRTblgenLib)

add_mlir_library_install(MLIRTblgenLib)
