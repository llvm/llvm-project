get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

if(MLIR_INCLUDE_TESTS)
  set(test_libs
    MLIRTestDialect
    )
endif()

add_mlir_tool(mlir-query
  mlir-query.cpp
  )
llvm_update_compile_flags(mlir-query)
mlir_target_link_libraries(mlir-query
  PRIVATE
  ${dialect_libs}
  MLIRQueryLib
  )
target_link_libraries(mlir-query PRIVATE ${test_libs})

mlir_check_link_libraries(mlir-query)
