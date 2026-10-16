set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

if(MLIR_INCLUDE_TESTS)
  set(test_libs
    ${cuda_test_libs}
    MLIRTestFuncToLLVM
    MLIRAffineTransformsTestPasses
    MLIRArithTestPasses
    MLIRArmSMETestPasses
    MLIRBufferizationTestPasses
    MLIRControlFlowTestPasses
    MLIRDLTITestPasses
    MLIRFuncTestPasses
    MLIRGPUTestPasses
    MLIRLinalgTestPasses
    MLIRLoopLikeInterfaceTestPasses
    MLIRMathTestPasses
    MLIRTestMathToVCIX
    MLIRMemRefTestPasses
    MLIRTestMemRefToLLVMWithTransforms
    MLIRShardTest
    MLIRNVGPUTestPasses
    MLIROpenACCTestPasses
    MLIRSCFTestPasses
    MLIRShapeTestPasses
    MLIRSPIRVTestPasses
    MLIRTensorTestPasses
    MLIRTestAnalysis
    MLIRTestConvertToSPIRV
    MLIRTestDialect
    MLIRTestDynDialect
    MLIRTestIR
    MLIRTestPass
    MLIRTestReducer
    MLIRTestTransforms
    MLIRTilingInterfaceTestPasses
    MLIRTosaTestPasses
    MLIRVectorTestPasses
    MLIRXeGPUTestPasses
    MLIRTestVectorToSPIRV
    MLIRLLVMTestPasses
    )
  set(test_libs ${test_libs}
    MLIRTestPDLL
    MLIRTestTransformDialect
    MLIRTestIRDLToCppDialect
    )

  if (MLIR_ENABLE_PDL_IN_PATTERNMATCH)
    set(test_libs ${test_libs}
      MLIRTestPDLL
      MLIRTestRewrite
      )
  endif()
endif()

set(LIBS

  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses

  # TODO: Remove when registerAllGPUToLLVMIRTranslations is no longer
  # registered directly in mlir-opt.cpp.
  MLIRToLLVMIRTranslationRegistration
  )

# Exclude from libMLIR.so because this has static options intended for
# opt-like tools only.
add_mlir_library(MLIRMlirOptMain
  mlir-opt.cpp

  EXCLUDE_FROM_LIBMLIR
  )
mlir_target_link_libraries(MLIRMlirOptMain PUBLIC
  ${LIBS}
  ${test_libs}
  )

add_mlir_tool(mlir-opt
  mlir-opt.cpp

  SUPPORT_PLUGINS
  )
mlir_target_link_libraries(mlir-opt PRIVATE ${LIBS})
target_link_libraries(mlir-opt PRIVATE ${test_libs})
llvm_update_compile_flags(mlir-opt)

mlir_check_all_link_libraries(mlir-opt)
export_executable_symbols_for_plugins(mlir-opt)
