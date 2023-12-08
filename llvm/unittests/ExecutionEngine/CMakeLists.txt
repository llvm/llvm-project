set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  Interpreter
  MC
  OrcJIT
  RuntimeDyld
  Support
  )

add_llvm_unittest(ExecutionEngineTests
  ExecutionEngineTest.cpp
  )

add_subdirectory(JITLink)
add_subdirectory(Orc)

# Include MCJIT tests only if native arch is a built JIT target.
list(FIND LLVM_TARGETS_TO_BUILD "${LLVM_NATIVE_ARCH}" build_idx)
list(FIND LLVM_TARGETS_WITH_JIT "${LLVM_NATIVE_ARCH}" jit_idx)
if (NOT build_idx LESS 0 AND NOT jit_idx LESS 0)
  add_subdirectory(MCJIT)
endif()

set_property(TARGET ExecutionEngineTests PROPERTY FOLDER "Tests/UnitTests/ExecutionTests")
