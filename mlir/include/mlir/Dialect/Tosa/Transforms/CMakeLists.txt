set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TosaOpt)
mlir_tablegen(PassesEnums.h.inc -gen-enum-decls)
mlir_tablegen(PassesEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRTosaPassIncGen)
add_dependencies(mlir-headers MLIRTosaPassIncGen)

add_mlir_doc(Passes TosaPasses ./ -gen-pass-doc)
