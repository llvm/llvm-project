function(add_irdl_to_cpp_target target irdl_file)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${irdl_file}.cpp.inc
    COMMAND ${MLIR_IRDL_TO_CPP_EXE} ${CMAKE_CURRENT_SOURCE_DIR}/${irdl_file} -o ${CMAKE_CURRENT_BINARY_DIR}/${irdl_file}.cpp.inc
    DEPENDS ${MLIR_IRDL_TO_CPP_TARGET} ${irdl_file}
    COMMENT "Building ${irdl_file}..."
  )
  add_custom_target(${target} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${irdl_file}.cpp.inc)
endfunction()
