include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_partially_linked_object(LLVMX86AsmPrinter
  X86ATTAsmPrinter.cpp
  X86AsmPrinter.cpp
  X86IntelAsmPrinter.cpp
  )

target_name_of_partially_linked_object(LLVMX86CodeGen n)

add_dependencies(LLVMX86AsmPrinter ${n})
