add_subdirectory(mlir-lsp-server)
add_subdirectory(mlir-opt)
add_subdirectory(mlir-parser-fuzzer)
add_subdirectory(mlir-pdll-lsp-server)
add_subdirectory(mlir-query)
add_subdirectory(mlir-reduce)
add_subdirectory(mlir-shlib)
add_subdirectory(mlir-spirv-cpu-runner)
add_subdirectory(mlir-translate)
add_subdirectory(mlir-vulkan-runner)
add_subdirectory(tblgen-lsp-server)
add_subdirectory(tblgen-to-irdl)

# mlir-cpu-runner requires ExecutionEngine.
if(MLIR_ENABLE_EXECUTION_ENGINE)
  add_subdirectory(mlir-cpu-runner)
endif()

# FIXME: Currently depends on utility functions inside clang.
if ("clang" IN_LIST LLVM_ENABLE_PROJECTS)
  add_subdirectory(mlir-rewrite)
endif()
