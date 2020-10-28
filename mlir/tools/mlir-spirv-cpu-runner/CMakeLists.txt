set(LLVM_OPTIONAL_SOURCES
  mlir-spirv-cpu-runner.cpp
  )

if (MLIR_SPIRV_CPU_RUNNER_ENABLED)
  message(STATUS "Building SPIR-V cpu runner")

  add_llvm_tool(mlir-spirv-cpu-runner
    mlir-spirv-cpu-runner.cpp
  )

  llvm_update_compile_flags(mlir-spirv-cpu-runner)

  get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
  get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

  target_link_libraries(mlir-spirv-cpu-runner PRIVATE
    ${conversion_libs}
    ${dialect_libs}
    MLIRAnalysis
    MLIREDSC
    MLIRExecutionEngine
    MLIRIR
    MLIRJitRunner
    MLIRLLVMIR
    MLIRParser
    MLIRTargetLLVMIR
    MLIRTransforms
    MLIRTranslation
    MLIRSupport
  )
endif()
