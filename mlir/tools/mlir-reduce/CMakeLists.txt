get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

if(MLIR_INCLUDE_TESTS)
  set(test_libs
    MLIRTestDialect
    )
endif()

set(LIBS
  ${conversion_libs}
  ${dialect_libs}
  MLIRDialect
  MLIRIR
  MLIRPass
  MLIRReduceLib
  )

add_mlir_tool(mlir-reduce
  mlir-reduce.cpp
  )

mlir_target_link_libraries(mlir-reduce PRIVATE ${LIBS})
target_link_libraries(mlir-reduce PRIVATE ${test_libs})
llvm_update_compile_flags(mlir-reduce)

mlir_check_all_link_libraries(mlir-reduce)
