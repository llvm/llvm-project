set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Mesh)
add_public_tablegen_target(MLIRMeshPassIncGen)
add_dependencies(mlir-headers MLIRMeshPassIncGen)

add_mlir_doc(Passes MeshPasses ./ -gen-pass-doc)
