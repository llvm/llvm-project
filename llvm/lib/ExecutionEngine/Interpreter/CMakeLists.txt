if( LLVM_ENABLE_FFI )
  set(link_libs FFI::ffi)
endif()

add_llvm_component_library(LLVMInterpreter
  Execution.cpp
  ExternalFunctions.cpp
  Interpreter.cpp

  DEPENDS
  intrinsics_gen

  LINK_LIBS PRIVATE ${link_libs}
  LINK_COMPONENTS
  CodeGen
  Core
  ExecutionEngine
  Support
  )
