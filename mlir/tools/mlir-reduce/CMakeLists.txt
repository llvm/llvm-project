get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
  ${conversion_libs}
  ${dialect_libs}
  MLIRDialect
  MLIRIR
  MLIRPass
  MLIRReduceLib
  MLIRTestDialect
  )

add_llvm_tool(mlir-reduce
  mlir-reduce.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(mlir-reduce PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-reduce)

mlir_check_all_link_libraries(mlir-reduce)
