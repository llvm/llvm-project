add_mlir_dialect(SMT smt)

set(LLVM_TARGET_DEFINITIONS SMT.td)

mlir_tablegen(SMTAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(SMTAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRSMTAttrIncGen)
add_dependencies(mlir-headers MLIRSMTAttrIncGen)

mlir_tablegen(SMTEnums.h.inc -gen-enum-decls)
mlir_tablegen(SMTEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRSMTEnumsIncGen)
add_dependencies(mlir-headers MLIRSMTEnumsIncGen)

add_mlir_doc(SMT SMT Dialects/ -gen-dialect-doc)
