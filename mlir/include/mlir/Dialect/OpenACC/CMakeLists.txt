set(LLVM_TARGET_DEFINITIONS ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend/OpenACC/ACC.td)
mlir_tablegen(AccCommon.td --gen-directive-decl --directives-dialect=OpenACC)
add_public_tablegen_target(acc_common_td)

add_mlir_dialect(OpenACCOps acc)

add_mlir_doc(OpenACCOps OpenACCDialect Dialects/ -gen-dialect-doc -dialect=acc)
add_dependencies(OpenACCDialectDocGen acc_common_td)

set(LLVM_TARGET_DEFINITIONS OpenACCOps.td)
mlir_tablegen(OpenACCOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpenACCOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIROpenACCEnumsIncGen)
add_dependencies(mlir-headers MLIROpenACCEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS OpenACCOps.td)
mlir_tablegen(OpenACCOpsAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=acc)
mlir_tablegen(OpenACCOpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=acc)
add_public_tablegen_target(MLIROpenACCAttributesIncGen)
add_dependencies(mlir-headers MLIROpenACCAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS OpenACCTypeInterfaces.td)
mlir_tablegen(OpenACCTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(OpenACCTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIROpenACCTypeInterfacesIncGen)
add_dependencies(mlir-headers MLIROpenACCTypeInterfacesIncGen)
