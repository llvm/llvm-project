# Declare a function to generate ODS with mlir-linalg-ods-yaml-gen
function(add_linalg_ods_yaml_gen yaml_ast_file output_file)
  set(YAML_AST_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/${yaml_ast_file})
  set(GEN_ODS_FILE ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.yamlgen.td)
  set(GEN_CPP_FILE ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.yamlgen.cpp.inc)
  set_source_files_properties(
    ${GEN_ODS_FILE}
    PROPERTIES GENERATED TRUE)
  set_source_files_properties(
    ${GEN_CPP_FILE}
    PROPERTIES GENERATED TRUE)
  add_custom_command(
    OUTPUT ${GEN_ODS_FILE} ${GEN_CPP_FILE}
    COMMAND ${MLIR_LINALG_ODS_YAML_GEN_EXE} ${YAML_AST_SOURCE} -o-ods-decl=${GEN_ODS_FILE} -o-impl=${GEN_CPP_FILE}
    MAIN_DEPENDENCY
    ${YAML_AST_SOURCE}
    DEPENDS
    ${MLIR_LINALG_ODS_YAML_GEN_EXE}
    ${MLIR_LINALG_ODS_YAML_GEN_TARGET})
  add_custom_target(
    MLIR${output_file}YamlIncGen
    DEPENDS
    ${MLIR_LINALG_ODS_YAML_GEN_EXE}
    ${MLIR_LINALG_ODS_YAML_GEN_TARGET}
    ${GEN_ODS_FILE} ${GEN_CPP_FILE})
  list(APPEND LLVM_TARGET_DEPENDS ${GEN_ODS_FILE})
  set(LLVM_TARGET_DEPENDS ${LLVM_TARGET_DEPENDS} PARENT_SCOPE)
endfunction()

# NOTE: LLVM_TARGET_DEPENDS gets picked up by tablegen targets to add file
# level dependencies. This is gross but CMake requires depending on both
# targets and generated files, and it must be done when the custom target is
# declared (there is no way to add after the fact).
set(LLVM_TARGET_DEPENDS "")
add_linalg_ods_yaml_gen(LinalgNamedStructuredOps.yaml LinalgNamedStructuredOps)

# Provide a short name for all external dependency that needs to
# include Linalg in ODS
add_custom_target(LinalgOdsGen
  DEPENDS
  MLIRLinalgNamedStructuredOpsYamlIncGen
)
add_dependencies(mlir-headers LinalgOdsGen)

add_mlir_dialect(LinalgOps linalg)

set(LLVM_TARGET_DEFINITIONS LinalgEnums.td)
mlir_tablegen(LinalgOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(LinalgOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRLinalgOpsEnumsIncGen)
add_dependencies(mlir-headers MLIRLinalgOpsEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS LinalgOps.td)
mlir_tablegen(LinalgOpsAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(LinalgOpsAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRLinalgOpsAttributesIncGen)
add_dependencies(mlir-headers MLIRLinalgOpsAttributesIncGen)

add_mlir_doc(LinalgDoc LinalgOps Dialects/ -gen-op-doc)
add_dependencies(LinalgOpsDocGen LinalgOdsGen)

set(LLVM_TARGET_DEFINITIONS LinalgStructuredOps.td)
mlir_tablegen(LinalgStructuredOps.h.inc -gen-op-decls)
mlir_tablegen(LinalgStructuredOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRLinalgStructuredOpsIncGen)
add_dependencies(MLIRLinalgStructuredOpsIncGen LinalgOdsGen)
add_dependencies(mlir-headers MLIRLinalgStructuredOpsIncGen)

set(LLVM_TARGET_DEFINITIONS LinalgInterfaces.td)
mlir_tablegen(LinalgInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(LinalgInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRLinalgInterfacesIncGen)
add_dependencies(mlir-headers MLIRLinalgInterfacesIncGen)
