add_subdirectory(mlir-lsp-server)
add_subdirectory(mlir-opt)
add_subdirectory(mlir-parser-fuzzer)
add_subdirectory(mlir-pdll)
add_subdirectory(mlir-pdll-lsp-server)
add_subdirectory(mlir-reduce)
add_subdirectory(mlir-shlib)
add_subdirectory(mlir-spirv-cpu-runner)
add_subdirectory(mlir-translate)
add_subdirectory(mlir-vulkan-runner)

# mlir-cpu-runner requires ExecutionEngine which is only built
# when the native target is configured in.
if(TARGET ${LLVM_NATIVE_ARCH})
  add_subdirectory(mlir-cpu-runner)
endif()
