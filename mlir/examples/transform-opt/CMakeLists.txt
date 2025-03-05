get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTransformDialect
  MLIRTransformDialectTransforms
  MLIRTransforms
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
)

add_mlir_tool(mlir-transform-opt
  mlir-transform-opt.cpp

  DEPENDS
  ${LIBS}
)
target_link_libraries(mlir-transform-opt PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-transform-opt)
mlir_check_all_link_libraries(mlir-transform-opt)
