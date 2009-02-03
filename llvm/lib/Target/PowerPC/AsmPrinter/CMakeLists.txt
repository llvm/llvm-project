include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_partially_linked_object(LLVMPowerPCAsmPrinter
  PPCAsmPrinter.cpp
  )

target_name_of_partially_linked_object(LLVMPowerPCCodeGen n)

add_dependencies(LLVMPowerPCAsmPrinter ${n})
