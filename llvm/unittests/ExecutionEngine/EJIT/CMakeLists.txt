set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  EmbeddedJIT
  EJIT
  ExecutionEngine
  OrcJIT
  Passes
  Support
  TargetParser
  )

add_llvm_unittest(EJITTests
  EJitRuntimeTest.cpp

  PARTIAL_SOURCES_INTENDED
  )

# SRE code pool tests live in their own executable so they build and run
# independently of EJITTests (host-runnable: mock allocator + mock enable_ex,
# no real SRE symbols required).
add_llvm_unittest(EJITCodePoolTests
  EJitCodePoolTest.cpp
  EJitCodePoolMemoryManagerTest.cpp

  PARTIAL_SOURCES_INTENDED
  )

# Convenience target: build + run only the SRE code pool unit tests.
add_custom_target(check-ejit-code-pool
  COMMAND $<TARGET_FILE:EJITCodePoolTests>
  DEPENDS EJITCodePoolTests
  COMMENT "Running EmbeddedJIT SRE code pool unit tests"
  USES_TERMINAL
  )
