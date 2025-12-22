set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Tosa)
mlir_tablegen(PassesEnums.h.inc -gen-enum-decls)
mlir_tablegen(PassesEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Tosa)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Tosa)
add_mlir_dialect_tablegen_target(MLIRTosaPassIncGen)

add_mlir_doc(Passes TosaPasses ./ -gen-pass-doc)
