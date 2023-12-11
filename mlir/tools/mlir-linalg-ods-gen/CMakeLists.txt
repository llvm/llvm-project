set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

set(LIBS
  MLIRIR
  MLIRSupport
  MLIRParser
)

# New mlir-linalg-ods-yaml-gen.
add_tablegen(mlir-linalg-ods-yaml-gen MLIR_LINALG_ODS_YAML_GEN
  DESTINATION "${MLIR_TOOLS_INSTALL_DIR}"
  EXPORT MLIR
  SKIP_COMPONENT_LINK
  mlir-linalg-ods-yaml-gen.cpp

  DEPENDS
  ${LIBS}
)
set_target_properties(mlir-linalg-ods-yaml-gen PROPERTIES FOLDER "Tablegenning")
target_link_libraries(mlir-linalg-ods-yaml-gen PRIVATE ${LIBS})

mlir_check_all_link_libraries(mlir-linalg-ods-yaml-gen)

configure_file(
  update_core_linalg_named_ops.sh.in
  ${MLIR_TOOLS_DIR}/update_core_linalg_named_ops.sh)
