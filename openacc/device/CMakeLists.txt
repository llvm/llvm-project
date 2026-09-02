# Ensure the compiler is a valid clang when building the GPU target.
set(req_ver "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")
if(LLVM_VERSION_MAJOR AND NOT (CMAKE_CXX_COMPILER_ID MATCHES "[Cc]lang" AND
   ${CMAKE_CXX_COMPILER_VERSION} VERSION_EQUAL "${req_ver}"))
  message(FATAL_ERROR "Cannot build OpenACC GPU device runtime. CMake compiler "
                      "'${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}' "
                      " is not 'Clang ${req_ver}'.")
endif()

set(src_files
  ${CMAKE_CURRENT_SOURCE_DIR}/src/AccEntryImpl.cpp
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

list(APPEND compile_options "SHELL: -mllvm -vectorize-slp=false")
if("${LLVM_DEFAULT_TARGET_TRIPLE}" MATCHES "^amdgcn" OR
   "${CMAKE_CXX_COMPILER_TARGET}" MATCHES "^amdgcn")
  set(target_name "amdgpu")
  list(APPEND compile_options "SHELL:-Xclang -mcode-object-version=none")
elseif("${LLVM_DEFAULT_TARGET_TRIPLE}" MATCHES "^nvptx" OR
       "${CMAKE_CXX_COMPILER_TARGET}" MATCHES "^nvptx")
  set(target_name "nvptx")
  list(APPEND compile_options --cuda-feature=+ptx63)
elseif("${LLVM_DEFAULT_TARGET_TRIPLE}" MATCHES "^spirv" OR
       "${CMAKE_CXX_COMPILER_TARGET}" MATCHES "^spirv")
  set(target_name "spirv")
endif()

# Produce a bitcode file via the linker's LTO pass.
add_executable(libllvmaccdevice ${src_files})
set_target_properties(libllvmaccdevice PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${LLVM_LIBRARY_OUTPUT_INTDIR}/${LLVM_DEFAULT_TARGET_TRIPLE}"
  LINKER_LANGUAGE CXX
  BUILD_RPATH ""
  INSTALL_RPATH ""
  RUNTIME_OUTPUT_NAME libacctarget-${target_name}.bc)

target_include_directories(libllvmaccdevice PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/include
                           ${CMAKE_CURRENT_SOURCE_DIR}/../../offload/libacctarget/include
                           ${CMAKE_CURRENT_SOURCE_DIR}/../../offload/include)
target_compile_options(libllvmaccdevice PRIVATE ${compile_options})
if(NOT "${LLVM_DEFAULT_TARGET_TRIPLE}" MATCHES "^spirv" AND
   NOT "${CMAKE_CXX_COMPILER_TARGET}" MATCHES "^spirv")
  target_link_options(libllvmaccdevice PRIVATE
                  "-flto" "-r" "-nostdlib" "-Wl,--lto-emit-llvm")
else()
  target_link_options(libllvmaccdevice PRIVATE
                  "-nostdlib" "-emit-llvm")
endif()

if(LLVM_DEFAULT_TARGET_TRIPLE)
  target_link_options(libllvmaccdevice PRIVATE "--target=${LLVM_DEFAULT_TARGET_TRIPLE}")
endif()
install(TARGETS libllvmaccdevice
        PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
        DESTINATION ${OPENACC_INSTALL_LIBDIR})

add_library(llvmaccdevice.all_objs OBJECT IMPORTED)
set_property(TARGET llvmaccdevice.all_objs APPEND PROPERTY IMPORTED_OBJECTS
             ${LLVM_LIBRARY_OUTPUT_INTDIR}/${LLVM_DEFAULT_TARGET_TRIPLE}/libacctarget-${target_name}.bc)

add_library(llvmaccdevice STATIC)
add_dependencies(llvmaccdevice libllvmaccdevice)
set_target_properties(llvmaccdevice PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${LLVM_LIBRARY_OUTPUT_INTDIR}/${LLVM_DEFAULT_TARGET_TRIPLE}"
  LINKER_LANGUAGE CXX
)
target_link_libraries(llvmaccdevice PRIVATE llvmaccdevice.all_objs)
install(TARGETS llvmaccdevice ARCHIVE DESTINATION "${OPENACC_INSTALL_LIBDIR}")
