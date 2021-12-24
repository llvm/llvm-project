set(LIBS
  MLIRPDLLAST
  MLIRPDLLParser
  )

add_llvm_tool(mlir-pdll
  mlir-pdll.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(mlir-pdll PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-pdll)

mlir_check_all_link_libraries(mlir-pdll)
