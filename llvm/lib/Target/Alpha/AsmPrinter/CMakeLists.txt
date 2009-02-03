include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_partially_linked_object(LLVMAlphaAsmPrinter
  AlphaAsmPrinter.cpp
  )

target_name_of_partially_linked_object(LLVMAlphaCodeGen n)

add_dependencies(LLVMAlphaAsmPrinter ${n})
