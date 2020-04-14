set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIRLoopOpsTransforms
  MLIRLoopAnalysis
  MLIRAffineTransformsTestPasses
  MLIRAnalysis
  MLIRDialect
  MLIREDSC
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRQuantizerFxpMathConfig
  MLIRQuantizerSupport
  MLIRQuantizerTransforms
  MLIRSPIRV
  MLIRSPIRVTestPasses
  MLIRSPIRVTransforms
  MLIRTransforms
  MLIRTransformUtils
  MLIRTestDialect
  MLIRTestIR
  MLIRTestPass
  MLIRTestTransforms
  MLIRSupport
  MLIRIR
  MLIROptLib
  LLVMSupport
  LLVMCore
  LLVMAsmParser
  )

add_llvm_library(MLIRMlirOptMain
  mlir-opt.cpp
)
target_link_libraries(MLIRMlirOptMain
  PUBLIC
  ${LIBS}
)

add_llvm_tool(mlir-opt
 mlir-opt.cpp
)

llvm_update_compile_flags(mlir-opt)
target_link_libraries(mlir-opt PRIVATE ${LIBS} ${targets_to_link})
