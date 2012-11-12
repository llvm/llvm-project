
set(LLVM_LINK_COMPONENTS mcjit jit interpreter nativecodegen bitreader asmparser selectiondag native)

if( LLVM_USE_OPROFILE )
  set(LLVM_LINK_COMPONENTS
    ${LLVM_LINK_COMPONENTS}
    OProfileJIT
    )
endif( LLVM_USE_OPROFILE )

if( LLVM_USE_INTEL_JITEVENTS )
  set(LLVM_LINK_COMPONENTS
    ${LLVM_LINK_COMPONENTS}
    IntelJITEvents
    )
endif( LLVM_USE_INTEL_JITEVENTS )

add_llvm_tool(lli
  lli.cpp
  RecordingMemoryManager.cpp
  RemoteTarget.cpp
  )
