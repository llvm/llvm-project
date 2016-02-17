include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/.. )

if( HAVE_LIBDL )
    set(LLVM_INTEL_JIT_LIBS ${CMAKE_DL_LIBS})
endif()
if( HAVE_LIBPTHREAD )
    set(LLVM_INTEL_JIT_LIBS pthread ${LLVM_INTEL_JIT_LIBS})
endif()

add_llvm_library(LLVMIntelJITEvents
  IntelJITEventListener.cpp
  jitprofiling.c

  LINK_LIBS ${LLVM_INTEL_JIT_LIBS}
)

add_dependencies(LLVMIntelJITEvents LLVMCodeGen)
