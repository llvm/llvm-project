set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  native
  )

add_mlir_tool(mlir-cpu-runner
  mlir-cpu-runner.cpp

  EXPORT_SYMBOLS
  )
llvm_update_compile_flags(mlir-cpu-runner)
mlir_target_link_libraries(mlir-cpu-runner PRIVATE
  MLIRAnalysis
  MLIRBuiltinToLLVMIRTranslation
  MLIRIR
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRToLLVMIRTranslationRegistration
  MLIRParser
  MLIRTargetLLVMIRExport
  MLIRSupport
  )
target_link_libraries(mlir-cpu-runner PRIVATE
  MLIRExecutionEngine
  MLIRJitRunner
  )
