set(LLVM_LINK_COMPONENTS Support)

add_tablegen(mlir-src-sharder MLIR_SRC_SHARDER
  DESTINATION "${MLIR_TOOLS_INSTALL_DIR}"
  EXPORT MLIR
  mlir-src-sharder.cpp)

set_target_properties(mlir-src-sharder PROPERTIES FOLDER "MLIR/Tablegenning")

mlir_check_all_link_libraries(mlir-src-sharder)
