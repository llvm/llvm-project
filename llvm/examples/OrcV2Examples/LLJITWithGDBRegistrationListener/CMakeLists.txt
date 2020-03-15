set(LLVM_LINK_COMPONENTS
  Core
  IRReader
  JITLink
  OrcJIT
  Support
  nativecodegen
  )

add_llvm_example(LLJITWithGDBRegistrationListener
  LLJITWithGDBRegistrationListener.cpp
  )

# We want JIT'd code to be able to link against process symbols like printf
# for this example, so make sure they're exported.
export_executable_symbols(LLJITWithGDBRegistrationListener)
