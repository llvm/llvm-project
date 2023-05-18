set(LIBS
  MLIRPdllLspServerLib
  )

add_mlir_tool(mlir-pdll-lsp-server
  mlir-pdll-lsp-server.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(mlir-pdll-lsp-server PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-pdll-lsp-server)

mlir_check_all_link_libraries(mlir-pdll-lsp-server)
