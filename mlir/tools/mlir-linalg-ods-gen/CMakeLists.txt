set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

# New mlir-linalg-ods-yaml-gen.
add_mlir_tool(mlir-linalg-ods-yaml-gen
  mlir-linalg-ods-yaml-gen.cpp
)
llvm_update_compile_flags(mlir-linalg-ods-yaml-gen)
target_link_libraries(mlir-linalg-ods-yaml-gen PRIVATE
  MLIRIR
  MLIRSupport
  MLIRParser
  )

setup_host_tool(mlir-linalg-ods-yaml-gen MLIR_LINALG_ODS_YAML_GEN MLIR_LINALG_ODS_YAML_GEN_EXE MLIR_LINALG_ODS_YAML_GEN_TARGET)

if(NOT ${MLIR_LINALG_ODS_YAML_GEN_EXE} STREQUAL "mlir-linalg-ods-yaml-gen")
  add_custom_target(mlir-linalg-ods-yaml-gen-host DEPENDS ${MLIR_LINALG_ODS_YAML_GEN_EXE})
  set_target_properties(mlir-linalg-ods-yaml-gen-host PROPERTIES FOLDER "MLIR/Tablegenning")

  if(NOT LLVM_BUILD_UTILS)
    set_target_properties(mlir-linalg-ods-yaml-gen PROPERTIES EXCLUDE_FROM_ALL ON)
  endif()
endif()

configure_file(
  update_core_linalg_named_ops.sh.in
  ${MLIR_TOOLS_DIR}/update_core_linalg_named_ops.sh)
