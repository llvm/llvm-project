include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_partially_linked_object(LLVMSparcAsmPrinter
  SparcAsmPrinter.cpp
  )

target_name_of_partially_linked_object(LLVMSparcCodeGen n)

add_dependencies(LLVMSparcAsmPrinter ${n})
