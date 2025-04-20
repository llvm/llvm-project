set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name DLTI)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix DLTI)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix DLTI)
add_public_tablegen_target(MLIRDLTIPassIncGen)

add_mlir_doc(Passes DLTIPasses ./ -gen-pass-doc)
