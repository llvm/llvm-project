set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  native
  )

add_mlir_tool(mlir-runner
  mlir-runner.cpp

  EXPORT_SYMBOLS
  )
llvm_update_compile_flags(mlir-runner)
mlir_target_link_libraries(mlir-runner PRIVATE
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
target_link_libraries(mlir-runner PRIVATE
  MLIRExecutionEngine
  MLIRJitRunner
  )

# Preventing re-export of symbols causes link errors with ASan and UBSan libs.
if (NOT LLVM_USE_SANITIZER)
  target_link_options(mlir-runner
    PRIVATE
      # On Linux, disable re-export of any static linked libraries that came
      # through. This prevents our LLVM build from interfering with the version
      # of LLVM included in certain graphics drivers.
      $<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs,ALL>
  )
endif()
