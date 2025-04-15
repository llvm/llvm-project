set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  ORCJIT
  Support
  Target
  nativecodegen
  )

# Enable EH and RTTI for this demo
if(NOT LLVM_ENABLE_EH)
  message(FATAL_ERROR "ExceptionDemo must require EH.")
endif()

add_llvm_example(ExceptionDemo
  ExceptionDemo.cpp

  EXPORT_SYMBOLS
  )
