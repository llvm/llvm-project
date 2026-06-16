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

# SRE taskpool tests: compile the taskpool sources directly with the taskpool
# macros so they build and run independently of LLVMEJIT / EJITTests and need
# no real SRE platform symbols (mock compiler + mock ring queue). Only LLVMSupport
# is linked (for the gtest harness), so this target never pulls in OrcJIT/EJIT.
set(EJIT_TASKPOOL_LINK_COMPONENTS_SAVED "${LLVM_LINK_COMPONENTS}")
set(LLVM_LINK_COMPONENTS Support)
add_llvm_unittest(EJITTaskPoolTests
  EJitTaskPoolTest.cpp
  ${LLVM_MAIN_SRC_DIR}/lib/ExecutionEngine/EJIT/EJitTaskPool.cpp
  ${LLVM_MAIN_SRC_DIR}/lib/ExecutionEngine/EJIT/EJitSreQueue.cpp

  PARTIAL_SOURCES_INTENDED
  )
set(LLVM_LINK_COMPONENTS "${EJIT_TASKPOOL_LINK_COMPONENTS_SAVED}")

target_compile_definitions(EJITTaskPoolTests PRIVATE
  EJIT_SRE_TASKPOOL
  EJIT_SRE_TASKPOOL_TESTING
  EJIT_SRE_TASKPOOL_BUCKETS=32
  EJIT_SRE_TASKPOOL_BUCKET_SLOTS=8
  EJIT_SRE_TASKPOOL_QUEUE_CAPACITY=1024
  )

# Convenience target: build + run only the SRE taskpool unit tests.
add_custom_target(check-ejit-taskpool
  COMMAND $<TARGET_FILE:EJITTaskPoolTests>
  DEPENDS EJITTaskPoolTests
  COMMENT "Running EmbeddedJIT SRE taskpool unit tests"
  USES_TERMINAL
  )
