set(LLVM_LINK_COMPONENTS
  CodeGenTypes
  Demangle
  Support
  TableGen
)

add_tablegen(mlir-tblgen MLIR
  DESTINATION "${MLIR_TOOLS_INSTALL_DIR}"
  EXPORT MLIR
  mlir-tblgen.cpp)

target_link_libraries(mlir-tblgen
  PRIVATE
  MLIRTblgenLib)

mlir_check_all_link_libraries(mlir-tblgen)
