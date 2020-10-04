set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  )

add_mlir_tool(mlir-cpu-runner
  mlir-cpu-runner.cpp
  )
llvm_update_compile_flags(mlir-cpu-runner)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(mlir-cpu-runner PRIVATE
  ${dialect_libs}
  MLIRAnalysis
  MLIREDSC
  MLIRExecutionEngine
  MLIRIR
  MLIRJitRunner
  MLIRLLVMIR
  MLIRParser
  MLIRTargetLLVMIR
  MLIRSupport
  )
