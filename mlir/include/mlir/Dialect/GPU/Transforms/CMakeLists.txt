set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name GPU)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix GPU)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix GPU)
add_mlir_dialect_tablegen_target(MLIRGPUPassIncGen)

add_mlir_doc(Passes GPUPasses ./ -gen-pass-doc)

add_subdirectory(NVVM)
add_subdirectory(ROCDL)
add_subdirectory(SPIRV)
add_subdirectory(XeVM)
