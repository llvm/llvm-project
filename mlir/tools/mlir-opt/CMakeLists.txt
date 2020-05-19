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

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIRLoopAnalysis
  MLIRAffineTransformsTestPasses
  MLIRAnalysis
  MLIRDialect
  MLIREDSC
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRSPIRVTestPasses
  MLIRTransforms
  MLIRTransformUtils
  MLIRTestDialect
  MLIRTestIR
  MLIRTestPass
  MLIRTestTransforms
  MLIRSupport
  MLIRIR
  )

# Exclude from libMLIR.so because this has static options intended for
# opt-like tools only.
add_mlir_library(MLIRMlirOptMain
  mlir-opt.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  ${LIBS}
  )

add_llvm_tool(mlir-opt
  mlir-opt.cpp

  DEPENDS
  ${LIBS}
  )
target_link_libraries(mlir-opt PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-opt)

mlir_check_all_link_libraries(mlir-opt)
