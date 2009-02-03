include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_partially_linked_object(LLVMARMAsmPrinter
  ARMAsmPrinter.cpp
  )

target_name_of_partially_linked_object(LLVMARMCodeGen n)

add_dependencies(LLVMARMAsmPrinter ${n})
