if ( LLVM_INCLUDE_UTILS )
  add_subdirectory(ChildTarget)
endif()

set(LLVM_LINK_COMPONENTS
  CodeGen
  Core
  ExecutionEngine
  IRReader
  Interpreter
  JITLink
  MC
  MCJIT
  Object
  OrcJIT
  OrcDebugging
  OrcShared
  OrcTargetProcess
  Passes
  RuntimeDyld
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
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
    DebugInfoDWARF
    IntelJITEvents
    Object
    )
endif( LLVM_USE_INTEL_JITEVENTS )

if( LLVM_USE_PERF )
  set(LLVM_LINK_COMPONENTS
    ${LLVM_LINK_COMPONENTS}
    DebugInfoDWARF
    PerfJITEvents
    Object
    )
endif( LLVM_USE_PERF )

if (MSVC AND CMAKE_MSVC_RUNTIME_LIBRARY AND
    (NOT CMAKE_MSVC_RUNTIME_LIBRARY MATCHES "DLL$"))
  # Do not export symbols when linking with the static CRT because in this case,
  # only a subset of CRT symbols is exported. When executing a module, this can
  # cause imported CRT symbols to resolve to different CRT instances. For
  # example, 'lli.exe' may export 'printf' but not 'fflush' or 'exit', causing
  # 'printf' to bind to the static CRT while the latter resolve to the dynamic
  # CRT. Disabling symbol export forces all imports to bind to the dynamic CRT
  # instance.
  set(MAYBE_EXPORT_SYMBOLS "")
else()
  set(MAYBE_EXPORT_SYMBOLS EXPORT_SYMBOLS)
endif()

add_llvm_tool(lli
  lli.cpp

  DEPENDS
  intrinsics_gen

  ${MAYBE_EXPORT_SYMBOLS}
  )
