set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

# New mlir-linalg-ods-yaml-gen.
add_llvm_tool(mlir-linalg-ods-yaml-gen
  mlir-linalg-ods-yaml-gen.cpp
)
llvm_update_compile_flags(mlir-linalg-ods-yaml-gen)
target_link_libraries(mlir-linalg-ods-yaml-gen PRIVATE
  MLIRIR
  MLIRSupport
  MLIRParser
  )

set(MLIR_LINALG_ODS_YAML_GEN mlir-linalg-ods-yaml-gen CACHE
  STRING "Native mlir-linalg-ods-yaml-gen executable. Saves building one when cross-compiling.")

set(MLIR_LINALG_ODS_YAML_GEN_EXE ${MLIR_LINALG_ODS_YAML_GEN} PARENT_SCOPE)
set(MLIR_LINALG_ODS_YAML_GEN_TARGET mlir-linalg-ods-yaml-gen PARENT_SCOPE)

if(LLVM_USE_HOST_TOOLS)
  if (${MLIR_LINALG_ODS_YAML_GEN} STREQUAL "mlir-linalg-ods-yaml-gen")
    build_native_tool(mlir-linalg-ods-yaml-gen MLIR_LINALG_ODS_YAML_GEN_EXE DEPENDS mlir-linalg-ods-yaml-gen)
    set(MLIR_LINALG_ODS_YAML_GEN_EXE ${MLIR_LINALG_ODS_YAML_GEN_EXE} PARENT_SCOPE)

    add_custom_target(mlir-linalg-ods-yaml-gen-host DEPENDS ${MLIR_LINALG_ODS_YAML_GEN_EXE})
    set(MLIR_LINALG_ODS_YAML_GEN_TARGET mlir-linalg-ods-yaml-gen-host DEPENDS PARENT_SCOPE)

    if(NOT LLVM_BUILD_UTILS)
      set_target_properties(mlir-linalg-ods-yaml-gen PROPERTIES EXCLUDE_FROM_ALL ON)
    endif()
  endif()
endif()

configure_file(
  update_core_linalg_named_ops.sh.in
  ${MLIR_TOOLS_DIR}/update_core_linalg_named_ops.sh)
