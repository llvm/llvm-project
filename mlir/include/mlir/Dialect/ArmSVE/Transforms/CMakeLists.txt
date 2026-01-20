set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ArmSVE)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix ArmSVE)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix ArmSVE)
add_mlir_dialect_tablegen_target(MLIRArmSVEPassIncGen)

add_mlir_doc(Passes ArmSVEPasses ./ -gen-pass-doc)
