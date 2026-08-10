set(LIBS
  TableGenLspServerLib
  )

add_mlir_tool(tblgen-lsp-server
  tblgen-lsp-server.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(tblgen-lsp-server PRIVATE ${LIBS})
llvm_update_compile_flags(tblgen-lsp-server)

mlir_check_all_link_libraries(tblgen-lsp-server)
