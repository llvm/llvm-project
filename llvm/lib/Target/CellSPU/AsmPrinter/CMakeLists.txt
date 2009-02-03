include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  )

add_partially_linked_object(LLVMCellSPUAsmPrinter
  SPUAsmPrinter.cpp
  )

target_name_of_partially_linked_object(LLVMCellSPUCodeGen n)

add_dependencies(LLVMCellSPUAsmPrinter ${n})
