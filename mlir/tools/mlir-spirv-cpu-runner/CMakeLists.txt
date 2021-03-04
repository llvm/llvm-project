set(LLVM_LINK_COMPONENTS
  Linker
)

if (MLIR_SPIRV_CPU_RUNNER_ENABLED)
  message(STATUS "Building SPIR-V CPU runner")

  add_llvm_tool(mlir-spirv-cpu-runner
    mlir-spirv-cpu-runner.cpp
  )

  llvm_update_compile_flags(mlir-spirv-cpu-runner)

  get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

  target_link_libraries(mlir-spirv-cpu-runner PRIVATE
    ${conversion_libs}
    MLIRAnalysis
    MLIREDSC
    MLIRExecutionEngine
    MLIRGPU
    MLIRIR
    MLIRJitRunner
    MLIRLLVMIR
    MLIRLLVMToLLVMIRTranslation
    MLIRParser
    MLIRSPIRV
    MLIRStandard
    MLIRTargetLLVMIRExport
    MLIRTransforms
    MLIRTranslation
    MLIRSupport
  )
endif()
