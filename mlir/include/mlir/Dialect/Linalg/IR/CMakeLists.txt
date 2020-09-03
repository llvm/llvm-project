# Declare a function to generate ODS with mlir-linalg-ods-gen
function(add_linalg_ods_gen tc_filename output_file)
  set(TC_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/${tc_filename})
  set(GEN_ODS_FILE ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.td)
  set(GEN_CPP_FILE ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.cpp.inc)
  set_source_files_properties(
    ${GEN_ODS_FILE}
    PROPERTIES GENERATED TRUE)
  set_source_files_properties(
    ${GEN_CPP_FILE}
    PROPERTIES GENERATED TRUE)
  add_custom_command(
    OUTPUT ${GEN_ODS_FILE} ${GEN_CPP_FILE}
    COMMAND mlir-linalg-ods-gen -gen-ods-decl ${TC_SOURCE} > ${GEN_ODS_FILE}
    COMMAND mlir-linalg-ods-gen -gen-impl ${TC_SOURCE} > ${GEN_CPP_FILE}
    MAIN_DEPENDENCY
    ${TC_SOURCE}
    DEPENDS
    mlir-linalg-ods-gen
    VERBATIM)
  add_custom_target(
    MLIR${output_file}IncGen
    DEPENDS
    mlir-linalg-ods-gen
    ${GEN_ODS_FILE} ${GEN_CPP_FILE})
endfunction()

add_linalg_ods_gen(LinalgNamedStructuredOpsSpec.tc LinalgNamedStructuredOps)
# Provide a short name for all external dependency that needs to
# include Linalg in ODS
add_custom_target(LinalgOdsGen DEPENDS MLIRLinalgNamedStructuredOpsIncGen)
add_dependencies(mlir-headers LinalgOdsGen)

add_mlir_dialect(LinalgOps linalg)

add_mlir_doc(LinalgDoc -gen-op-doc LinalgOps Dialects/)
add_dependencies(LinalgOpsDocGen LinalgOdsGen)

set(LLVM_TARGET_DEFINITIONS LinalgStructuredOps.td)
mlir_tablegen(LinalgStructuredOps.h.inc -gen-op-decls)
mlir_tablegen(LinalgStructuredOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRLinalgStructuredOpsIncGen)
add_dependencies(MLIRLinalgStructuredOpsIncGen LinalgOdsGen)
add_dependencies(mlir-headers MLIRLinalgStructuredOpsIncGen)

set(LLVM_TARGET_DEFINITIONS LinalgStructuredOpsInterface.td)
mlir_tablegen(LinalgStructuredOpsInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(LinalgStructuredOpsInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRLinalgStructuredOpsInterfaceIncGen)
add_dependencies(mlir-headers MLIRLinalgStructuredOpsInterfaceIncGen)
