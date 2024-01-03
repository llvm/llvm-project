set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

if(MLIR_INCLUDE_TESTS)
  set(test_libs
    MLIRAffineTransformsTestPasses
    MLIRShapeTestPasses
    MLIRSPIRVTestPasses
    MLIRTestAnalysis
    MLIRTestDialect
    MLIRTestDynDialect
    MLIRTestIR
    MLIRTestPass
    MLIRTestReducer
    MLIRTestRewrite
    MLIRTestTransformDialect
    MLIRTestTransforms
    )
endif()

set(LIBS
  ${conversion_libs}
  ${dialect_libs}
  ${extension_libs}
  ${test_libs}

  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIRFuncAllExtensions
  MLIRLspServerLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  )

add_mlir_tool(mlir-lsp-server
  mlir-lsp-server.cpp

  DEPENDS
  ${LIBS}
  )
target_link_libraries(mlir-lsp-server PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-lsp-server)

mlir_check_all_link_libraries(mlir-lsp-server)
