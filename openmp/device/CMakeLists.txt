# Ensure the compiler is a valid clang when building the GPU target.
set(req_ver "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")
if(LLVM_VERSION_MAJOR AND NOT (CMAKE_CXX_COMPILER_ID MATCHES "[Cc]lang" AND
   ${CMAKE_CXX_COMPILER_VERSION} VERSION_EQUAL "${req_ver}"))
  message(FATAL_ERROR "Cannot build GPU device runtime. CMake compiler "
                      "'${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}' "
                      " is not 'Clang ${req_ver}'.")
endif()

set(src_files
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Allocator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Configuration.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Debug.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Kernel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/LibC.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Mapping.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Misc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Parallelism.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Profiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Reduction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/State.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Synchronization.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Tasking.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/DeviceUtils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Workshare.cpp
)

list(APPEND compile_options -flto)
list(APPEND compile_options -fvisibility=hidden)
list(APPEND compile_options -nogpulib)
list(APPEND compile_options -nostdlibinc)
list(APPEND compile_options -fno-rtti)
list(APPEND compile_options -fno-exceptions)
list(APPEND compile_options -fconvergent-functions)
list(APPEND compile_options -Wno-unknown-cuda-version)
if(LLVM_DEFAULT_TARGET_TRIPLE)
  list(APPEND compile_options --target=${LLVM_DEFAULT_TARGET_TRIPLE})
endif()

# We disable the slp vectorizer during the runtime optimization to avoid
# vectorized accesses to the shared state. Generally, those are "good" but
# the optimizer pipeline (esp. Attributor) does not fully support vectorized
# instructions yet and we end up missing out on way more important constant
# propagation. That said, we will run the vectorizer again after the runtime
# has been linked into the user program.
list(APPEND compile_options "SHELL: -mllvm -vectorize-slp=false")
if("${LLVM_DEFAULT_TARGET_TRIPLE}" MATCHES "^amdgcn" OR
   "${CMAKE_CXX_COMPILER_TARGET}" MATCHES "^amdgcn")
  set(target_name "amdgpu")
  list(APPEND compile_options "SHELL:-Xclang -mcode-object-version=none")
elseif("${LLVM_DEFAULT_TARGET_TRIPLE}" MATCHES "^nvptx" OR
       "${CMAKE_CXX_COMPILER_TARGET}" MATCHES "^nvptx")
  set(target_name "nvptx")
  list(APPEND compile_options --cuda-feature=+ptx63)
endif()

# Trick to combine these into a bitcode file via the linker's LTO pass.
add_executable(libompdevice ${src_files})
set_target_properties(libompdevice PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  LINKER_LANGUAGE CXX
  BUILD_RPATH ""
  INSTALL_RPATH ""
  RUNTIME_OUTPUT_NAME libomptarget-${target_name}.bc)

# If the user built with the GPU C library enabled we will use that instead.
if(LIBOMPTARGET_GPU_LIBC_SUPPORT)
  target_compile_definitions(libompdevice PRIVATE OMPTARGET_HAS_LIBC)
endif()
target_compile_definitions(libompdevice PRIVATE SHARED_SCRATCHPAD_SIZE=512)

target_include_directories(libompdevice PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/include
                           ${CMAKE_CURRENT_SOURCE_DIR}/../../libc
                           ${CMAKE_CURRENT_SOURCE_DIR}/../../offload/include)
target_compile_options(libompdevice PRIVATE ${compile_options})
target_link_options(libompdevice PRIVATE
                    "-flto" "-r" "-nostdlib" "-Wl,--lto-emit-llvm")
if(LLVM_DEFAULT_TARGET_TRIPLE)
  target_link_options(libompdevice PRIVATE "--target=${LLVM_DEFAULT_TARGET_TRIPLE}")
endif()
install(TARGETS libompdevice
        PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
        DESTINATION ${OPENMP_INSTALL_LIBDIR})

add_library(ompdevice.all_objs OBJECT IMPORTED)
set_property(TARGET ompdevice.all_objs APPEND PROPERTY IMPORTED_OBJECTS
             ${CMAKE_CURRENT_BINARY_DIR}/libomptarget-${target_name}.bc)

# Archive all the object files generated above into a static library
add_library(ompdevice STATIC)
add_dependencies(ompdevice libompdevice)
set_target_properties(ompdevice PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${LLVM_LIBRARY_OUTPUT_INTDIR}/${LLVM_DEFAULT_TARGET_TRIPLE}"
  LINKER_LANGUAGE CXX
)
target_link_libraries(ompdevice PRIVATE ompdevice.all_objs)
install(TARGETS ompdevice ARCHIVE DESTINATION "${OPENMP_INSTALL_LIBDIR}")
