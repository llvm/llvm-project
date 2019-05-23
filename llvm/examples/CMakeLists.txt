add_subdirectory(BrainF)
add_subdirectory(Fibonacci)
add_subdirectory(HowToUseJIT)
add_subdirectory(HowToUseLLJIT)
add_subdirectory(Kaleidoscope)
add_subdirectory(ModuleMaker)

if(LLVM_ENABLE_EH AND (NOT WIN32) AND (NOT "${LLVM_NATIVE_ARCH}" STREQUAL "ARM"))
    add_subdirectory(ExceptionDemo)
endif()

if( HAVE_PTHREAD_H )
  add_subdirectory(ParallelJIT)
endif( HAVE_PTHREAD_H )
