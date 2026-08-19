set(LLVM_OFFLOAD_KERNEL_INCLUDE_DIRS
  ${CMAKE_CURRENT_BINARY_DIR}/../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/cuda
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hip
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/kernel
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../liboffload/include
  ${CMAKE_CURRENT_BINARY_DIR}/../../liboffload/API
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../plugins-nextgen/common/include)

function(add_llvm_offload_kernel_language_runtime_objects target language)
  add_library(${target} OBJECT
    src/LanguageRuntime.cpp
    src/LanguageErrors.cpp
  )
  add_dependencies(${target} OffloadAPI)
  target_include_directories(${target} PRIVATE ${LLVM_OFFLOAD_KERNEL_INCLUDE_DIRS})
  target_compile_options(${target} PRIVATE ${offload_compile_flags})
  target_compile_definitions(${target} PRIVATE
    LANGUAGE=${language}
    TARGET_NAME="libLLVMOffloadKernel"
    DEBUG_PREFIX="LLVMOffloadKernel"
  )
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

add_llvm_offload_kernel_language_runtime_objects(
  LLVMOffloadKernelCudaRuntimeObjects cuda)
add_llvm_offload_kernel_language_runtime_objects(
  LLVMOffloadKernelHipRuntimeObjects hip)

add_library(LLVMOffloadKernelPerThreadDefaultStreamObject OBJECT
  src/PerThreadDefaultStream.cpp
)
target_compile_options(LLVMOffloadKernelPerThreadDefaultStreamObject
  PRIVATE ${offload_compile_flags})
set_target_properties(LLVMOffloadKernelPerThreadDefaultStreamObject PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

set(LLVM_OFFLOAD_KERNEL_PER_THREAD_DEFAULT_STREAM_OBJECT
  "${LLVM_LIBRARY_OUTPUT_INTDIR}/${OFFLOAD_TARGET_SUBDIR}/LLVMOffloadKernelPerThreadDefaultStream.o")
add_custom_command(
  OUTPUT ${LLVM_OFFLOAD_KERNEL_PER_THREAD_DEFAULT_STREAM_OBJECT}
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "${LLVM_LIBRARY_OUTPUT_INTDIR}/${OFFLOAD_TARGET_SUBDIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_OBJECTS:LLVMOffloadKernelPerThreadDefaultStreamObject>
          ${LLVM_OFFLOAD_KERNEL_PER_THREAD_DEFAULT_STREAM_OBJECT}
  DEPENDS
          LLVMOffloadKernelPerThreadDefaultStreamObject
          $<TARGET_OBJECTS:LLVMOffloadKernelPerThreadDefaultStreamObject>
  VERBATIM)
add_custom_target(LLVMOffloadKernelPerThreadDefaultStream
  DEPENDS ${LLVM_OFFLOAD_KERNEL_PER_THREAD_DEFAULT_STREAM_OBJECT})

add_llvm_library(
  LLVMOffloadKernel SHARED

  $<TARGET_OBJECTS:LLVMOffloadKernelCudaRuntimeObjects>
  $<TARGET_OBJECTS:LLVMOffloadKernelHipRuntimeObjects>
  src/LanguageCommon.cpp
  src/LanguageLaunch.cpp
  src/LanguageRegistration.cpp
  src/State.cpp
  src/Stream.cpp
  )

if(LLVM_LINK_LLVM_DYLIB)
  set(llvm_libs LLVM)
else()
  llvm_map_components_to_libnames(llvm_libs Support)
endif()
target_link_libraries(LLVMOffloadKernel PRIVATE ${llvm_libs} LLVMOffload)
add_dependencies(LLVMOffloadKernel LLVMOffloadKernelPerThreadDefaultStream)

if(LLVM_HAVE_LINK_VERSION_SCRIPT)
  target_link_libraries(LLVMOffloadKernel PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")
endif()

target_include_directories(LLVMOffloadKernel PUBLIC
                            ${LLVM_OFFLOAD_KERNEL_INCLUDE_DIRS})

target_compile_options(LLVMOffloadKernel PRIVATE ${offload_compile_flags})
target_link_options(LLVMOffloadKernel PRIVATE ${offload_link_flags})

target_compile_definitions(LLVMOffloadKernel PRIVATE
  TARGET_NAME="libLLVMOffloadKernel"
  DEBUG_PREFIX="LLVMOffloadKernel"
)

set_target_properties(LLVMOffloadKernel PROPERTIES
                      RUNTIME_OUTPUT_DIRECTORY "${LLVM_LIBRARY_OUTPUT_INTDIR}/${OFFLOAD_TARGET_SUBDIR}"
                      POSITION_INDEPENDENT_CODE ON
                      INSTALL_RPATH "$ORIGIN"
                      BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")
install(TARGETS LLVMOffloadKernel
        COMPONENT offload
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        LIBRARY DESTINATION "${OFFLOAD_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${OFFLOAD_INSTALL_LIBDIR}")
install(FILES ${LLVM_OFFLOAD_KERNEL_PER_THREAD_DEFAULT_STREAM_OBJECT}
        DESTINATION "${OFFLOAD_INSTALL_LIBDIR}"
        COMPONENT offload)

install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/kernel/DefineLanguageNames.inc
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/kernel/LanguageErrors.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/kernel/LanguageMath.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/kernel/LanguageRuntime.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/kernel/UndefineLanguageNames.inc
        ${CMAKE_CURRENT_SOURCE_DIR}/include/LanguageLaunch.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Stream.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Types.h
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/offload/kernel/)
