add_linalg_ods_yaml_gen(LinalgNamedStructuredOps.yaml LinalgNamedStructuredOps)
add_public_tablegen_target(MLIRLinalgNamedStructuredOpsYamlIncGen)

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
