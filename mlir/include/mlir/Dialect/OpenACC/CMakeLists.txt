add_subdirectory(Transforms)

set(LLVM_TARGET_DEFINITIONS ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend/OpenACC/ACC.td)
mlir_tablegen(AccCommon.td --gen-directive-decl --directives-dialect=OpenACC)
add_mlir_dialect_tablegen_target(acc_common_td)

add_mlir_dialect(OpenACCOps acc)

add_mlir_doc(OpenACCOps OpenACCDialectOps Dialects/ -gen-dialect-doc -dialect=acc)
add_dependencies(OpenACCDialectOpsDocGen acc_common_td)

set(LLVM_TARGET_DEFINITIONS OpenACCOps.td)
mlir_tablegen(OpenACCOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpenACCOpsEnums.cpp.inc -gen-enum-defs)
add_mlir_dialect_tablegen_target(MLIROpenACCEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS OpenACCOps.td)
mlir_tablegen(OpenACCOpsAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=acc)
mlir_tablegen(OpenACCOpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=acc)
add_mlir_dialect_tablegen_target(MLIROpenACCAttributesIncGen)

add_mlir_interface(OpenACCOpsInterfaces)

set(LLVM_TARGET_DEFINITIONS OpenACCTypeInterfaces.td)
mlir_tablegen(OpenACCTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(OpenACCTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_mlir_dialect_tablegen_target(MLIROpenACCTypeInterfacesIncGen)
