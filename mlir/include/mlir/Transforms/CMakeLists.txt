
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transforms)
mlir_tablegen(Transforms.capi.h.inc -gen-pass-capi-header --prefix Transforms)
mlir_tablegen(Transforms.capi.cpp.inc -gen-pass-capi-impl --prefix Transforms)
add_mlir_dialect_tablegen_target(MLIRTransformsPassIncGen)

set(LLVM_TARGET_DEFINITIONS DialectInlinerInterface.td)
mlir_tablegen(DialectInlinerInterface.h.inc -gen-dialect-interface-decls)
add_mlir_dialect_tablegen_target(MLIRTransformsDialectInterfaceIncGen)

add_mlir_doc(Passes GeneralPasses ./ -gen-pass-doc)
