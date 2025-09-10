set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ArmSME)
mlir_tablegen(PassesEnums.h.inc -gen-enum-decls)
mlir_tablegen(PassesEnums.cpp.inc -gen-enum-defs)
add_mlir_dialect_tablegen_target(MLIRArmSMETransformsIncGen)

add_mlir_doc(Passes ArmSMEPasses ./ -gen-pass-doc)
