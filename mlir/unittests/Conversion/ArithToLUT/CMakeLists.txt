if(MLIR_ENABLE_EXECUTION_ENGINE)
  set(LLVM_LINK_COMPONENTS
    nativecodegen
    native
    orcjit
    support
  )

  add_mlir_unittest(MLIRArithToLUTTests
    ArithToLUTTest.cpp
  )

  mlir_target_link_libraries(MLIRArithToLUTTests
    PRIVATE
    MLIRArithToLUT
    MLIRArithToLLVM
    MLIRFuncToLLVM
    MLIRMemRefToLLVM
    MLIRReconcileUnrealizedCasts
    MLIRRegisterAllDialects
  )

  target_link_libraries(MLIRArithToLUTTests
    PRIVATE
    MLIRExecutionEngine
  )
endif()
