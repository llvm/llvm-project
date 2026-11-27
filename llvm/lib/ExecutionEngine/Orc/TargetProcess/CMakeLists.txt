if( CMAKE_HOST_UNIX AND HAVE_LIBRT )
  set(rt_lib rt)
endif()

set(intel_jit_profiling )
if( LLVM_USE_INTEL_JITEVENTS )
  set(intel_jit_profiling IntelJITProfiling)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../IntelJITProfiling)
  if(NOT DEFINED ITTAPI_SOURCE_DIR)
      set(ITTAPI_SOURCE_DIR ${PROJECT_BINARY_DIR})
  endif()
  include_directories(${ITTAPI_SOURCE_DIR}/ittapi/include/ )
endif()

add_llvm_component_library(LLVMOrcTargetProcess
  ExecutorSharedMemoryMapperService.cpp
  DefaultHostBootstrapValues.cpp
  ExecutorResolver.cpp
  LibraryResolver.cpp
  JITLoaderGDB.cpp
  JITLoaderPerf.cpp
  JITLoaderVTune.cpp
  LibraryScanner.cpp
  OrcRTBootstrap.cpp
  RegisterEHFrames.cpp
  SimpleExecutorDylibManager.cpp
  SimpleExecutorMemoryManager.cpp
  SimpleRemoteEPCServer.cpp
  TargetExecutionUtils.cpp
  UnwindInfoManager.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ExecutionEngine/Orc/TargetProcess

  LINK_LIBS
  ${LLVM_PTHREAD_LIB}
  ${rt_lib}

  LINK_COMPONENTS
  ${intel_jit_profiling}
  BinaryFormat
  Object
  OrcShared
  Support
  TargetParser
  )

if ((MINGW OR CYGWIN) AND BUILD_SHARED_LIBS)
  # This library contains symbols marked with LLVM_ALWAYS_EXPORT (__declspec(
  # dllexport)) so that they can be found via GetProcAddress from a statically
  # linked exe. When any symbol in a DLL carries dllexport, the MinGW/Cygwin
  # linker switches to exclusive-export mode and omits all other symbols from
  # the export table. Add --export-all-symbols to restore full export of all
  # symbols despite the targeted dllexports.
  target_link_options(LLVMOrcTargetProcess PRIVATE LINKER:--export-all-symbols)
endif()
