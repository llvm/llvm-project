include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/.. )

set(GIT_REPOSITORY https://github.com/intel/ittapi.git)
set(GIT_TAG v3.18.8)

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ittapi)
    execute_process(COMMAND ${GIT_EXECUTABLE} clone ${GIT_REPOSITORY}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE GIT_CLONE_RESULT)
    if(NOT GIT_CLONE_RESULT EQUAL "0")
        message(FATAL_ERROR "git clone ${GIT_REPOSITORY} failed with ${GIT_CLONE_RESULT}, please clone ${GIT_REPOSITORY}")
    endif()
endif()

execute_process(COMMAND ${GIT_EXECUTABLE} checkout ${GIT_TAG}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ittapi
                RESULT_VARIABLE GIT_CHECKOUT_RESULT)
if(NOT GIT_CHECKOUT_RESULT EQUAL "0")
    message(FATAL_ERROR "git checkout ${GIT_TAG} failed with ${GIT_CHECKOUT_RESULT}, please checkout ${GIT_TAG} at ${CMAKE_CURRENT_SOURCE_DIR}/ittapi")
endif()

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/ittapi/include/ )

if( HAVE_LIBDL )
    set(LLVM_INTEL_JIT_LIBS ${CMAKE_DL_LIBS})
endif()

set(LLVM_INTEL_JIT_LIBS ${LLVM_PTHREAD_LIB} ${LLVM_INTEL_JIT_LIBS})


add_llvm_component_library(LLVMIntelJITEvents
  IntelJITEventListener.cpp
  jitprofiling.c
  ittapi/src/ittnotify/ittnotify_static.c

  LINK_LIBS ${LLVM_INTEL_JIT_LIBS}

  LINK_COMPONENTS
  CodeGen
  Core
  DebugInfoDWARF
  Support
  Object
  ExecutionEngine
)

add_dependencies(LLVMIntelJITEvents LLVMCodeGen)
