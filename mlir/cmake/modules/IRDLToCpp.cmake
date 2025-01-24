function(add_irdl_to_cpp_target target irdl_file)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${irdl_file}.cpp.inc
    COMMAND $<TARGET_FILE:mlir-irdl-to-cpp> ${CMAKE_CURRENT_SOURCE_DIR}/${irdl_file} -o ${CMAKE_CURRENT_BINARY_DIR}/${irdl_file}.cpp.inc
    DEPENDS mlir-irdl-to-cpp ${irdl_file}
    COMMENT "Building ${irdl_file}..."
  )
  add_custom_target(${target} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${irdl_file}.cpp.inc)
endfunction()
