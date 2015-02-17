add_subdirectory(ChildTarget)

set(LLVM_LINK_COMPONENTS
  CodeGen
  Core
  ExecutionEngine
  IRReader
  Instrumentation
  Interpreter
  MC
  MCJIT
  Object
  RuntimeDyld
  SelectionDAG
  Support
  native
  )

if( LLVM_USE_OPROFILE )
  set(LLVM_LINK_COMPONENTS
    ${LLVM_LINK_COMPONENTS}
    OProfileJIT
    )
endif( LLVM_USE_OPROFILE )

if( LLVM_USE_INTEL_JITEVENTS )
  set(LLVM_LINK_COMPONENTS
    ${LLVM_LINK_COMPONENTS}
    DebugInfo
    IntelJITEvents
    Object
    )
endif( LLVM_USE_INTEL_JITEVENTS )

add_llvm_tool(lli
  lli.cpp
  RemoteMemoryManager.cpp
  RemoteTarget.cpp
  RemoteTargetExternal.cpp
  )
set_target_properties(lli PROPERTIES ENABLE_EXPORTS 1)
