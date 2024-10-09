set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Mesh)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Mesh)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Mesh)
add_public_tablegen_target(MLIRMeshPassIncGen)
add_dependencies(mlir-headers MLIRMeshPassIncGen)

add_mlir_doc(Passes MeshPasses ./ -gen-pass-doc)
