set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
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
    MLIRMeshTest
    MLIRNVGPUTestPasses
    MLIRSCFTestPasses
    MLIRShapeTestPasses
    MLIRSPIRVTestPasses
    MLIRTensorTestPasses
    MLIRTestAnalysis
    MLIRTestDialect
    MLIRTestDynDialect
    MLIRTestIR
    MLIRTestOneToNTypeConversionPass
    MLIRTestPass
    MLIRTestReducer
    MLIRTestTransforms
    MLIRTilingInterfaceTestPasses
    MLIRVectorTestPasses
    MLIRTestVectorToSPIRV
    MLIRLLVMTestPasses
    )
  set(test_libs ${test_libs}
    MLIRTestPDLL
    MLIRTestTransformDialect
    )

  if (MLIR_ENABLE_PDL_IN_PATTERNMATCH)
    set(test_libs ${test_libs}
      MLIRTestPDLL
      MLIRTestRewrite
      )
  endif()
endif()

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  ${test_libs}

  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRCastInterfaces
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR

  # TODO: Remove when registerAllGPUToLLVMIRTranslations is no longer
  # registered directly in mlir-opt.cpp.
  MLIRToLLVMIRTranslationRegistration
  )

# Exclude from libMLIR.so because this has static options intended for
# opt-like tools only.
add_mlir_library(MLIRMlirOptMain
  mlir-opt.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  ${LIBS}
  )

add_mlir_tool(mlir-opt
  mlir-opt.cpp

  DEPENDS
  ${LIBS}
  SUPPORT_PLUGINS
  )
target_link_libraries(mlir-opt PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-opt)

mlir_check_all_link_libraries(mlir-opt)
export_executable_symbols_for_plugins(mlir-opt)

if(MLIR_ENABLE_DEPRECATED_GPU_SERIALIZATION)
  # Enable deprecated serialization passes.
  target_compile_definitions(mlir-opt
    PRIVATE
    MLIR_DEPRECATED_GPU_SERIALIZATION_ENABLE=1
  )
endif()
