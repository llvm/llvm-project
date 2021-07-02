# Declare a function to generate ODS with mlir-linalg-ods-gen
function(add_linalg_ods_tc_gen tc_filename output_file)
  set(TC_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/${tc_filename})
  set(GEN_ODS_FILE ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.tcgen.td)
  set(GEN_CPP_FILE ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.tcgen.cpp.inc)
  set_source_files_properties(
    ${GEN_ODS_FILE}
    PROPERTIES GENERATED TRUE)
  set_source_files_properties(
    ${GEN_CPP_FILE}
    PROPERTIES GENERATED TRUE)
  add_custom_command(
    OUTPUT ${GEN_ODS_FILE} ${GEN_CPP_FILE}
    COMMAND ${MLIR_LINALG_ODS_GEN_EXE} -gen-ods-decl ${TC_SOURCE} > ${GEN_ODS_FILE}
    COMMAND ${MLIR_LINALG_ODS_GEN_EXE} -gen-impl ${TC_SOURCE} > ${GEN_CPP_FILE}
    MAIN_DEPENDENCY
    ${TC_SOURCE}
    DEPENDS
    ${MLIR_LINALG_ODS_GEN_EXE}
    ${MLIR_LINALG_ODS_GEN_TARGET}
    VERBATIM)
  add_custom_target(
    MLIR${output_file}TcIncGen
    DEPENDS
    ${MLIR_LINALG_ODS_GEN_EXE}
    ${MLIR_LINALG_ODS_GEN_TARGET}
    ${GEN_ODS_FILE} ${GEN_CPP_FILE})
  # Setup the file dependencies needed for the subsequent tablegen step.
  # TODO: Once there is only one way of generating named ops remove this parent
  # scope manipulation and implement the tablegen generation in the same scope.
  set(LLVM_TARGET_DEPENDS ${LLVM_TARGET_DEPENDS} ${GEN_ODS_FILE} PARENT_SCOPE)
endfunction()

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
  # Setup the file dependencies needed for the subsequent tablegen step.
  # TODO: Once there is only one way of generating named ops remove this parent
  # scope manipulation and implement the tablegen generation in the same scope.
  set(LLVM_TARGET_DEPENDS ${LLVM_TARGET_DEPENDS} ${GEN_ODS_FILE} PARENT_SCOPE)
endfunction()

# TODO: Delete tc generation and replace with the YAML variant once all ops are
# ported. At the same time, move the YAML and TableGen generation to the same
# scope to avoid the at a distance dependency manipulation via
# LLVM_TARGET_DEPENDS.
set(LLVM_TARGET_DEPENDS "")
add_linalg_ods_tc_gen(LinalgNamedStructuredOpsSpec.tc LinalgNamedStructuredOps)
add_linalg_ods_yaml_gen(LinalgNamedStructuredOps.yaml LinalgNamedStructuredOps)

# Provide a short name for all external dependency that needs to
# include Linalg in ODS
add_custom_target(LinalgOdsGen
  DEPENDS
  MLIRLinalgNamedStructuredOpsTcIncGen
  MLIRLinalgNamedStructuredOpsYamlIncGen
)
add_dependencies(mlir-headers LinalgOdsGen)

add_mlir_dialect(LinalgOps linalg)

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
