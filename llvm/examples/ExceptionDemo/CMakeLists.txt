set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  MCJIT
  Support
  nativecodegen
  )

# Enable EH and RTTI for this demo
set(LLVM_REQUIRES_EH 1)
set(LLVM_REQUIRES_RTTI 1)

add_llvm_example(ExceptionDemo
  ExceptionDemo.cpp
  )

set_target_properties(ExceptionDemo PROPERTIES ENABLE_EXPORTS 1)
