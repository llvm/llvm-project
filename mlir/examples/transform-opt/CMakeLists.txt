set(LIBS
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRSupport
  MLIRTransformDialect
  MLIRTransformDialectTransforms
  MLIRTransforms
)

add_mlir_tool(mlir-transform-opt
  mlir-transform-opt.cpp

  DEPENDS
  ${LIBS}
)
target_link_libraries(mlir-transform-opt PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-transform-opt)
mlir_check_all_link_libraries(mlir-transform-opt)
