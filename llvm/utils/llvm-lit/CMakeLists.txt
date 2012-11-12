configure_file(
  llvm-lit.in
  ${LLVM_TOOLS_BINARY_DIR}/llvm-lit
  )

install(FILES
  ${LLVM_TOOLS_BINARY_DIR}/llvm-lit
  DESTINATION bin
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
              GROUP_READ GROUP_EXECUTE
              WORLD_READ WORLD_EXECUTE
  )
