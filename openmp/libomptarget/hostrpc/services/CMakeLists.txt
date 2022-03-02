
find_package(hsa-runtime64 1.2.0 REQUIRED HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)
add_library(hostrpc_services STATIC hostcall.cpp hostrpc_execute_service.c hostrpc_externs.c urilocator.cpp devsanitizer.cpp)
set_property(TARGET hostrpc_services PROPERTY POSITION_INDEPENDENT_CODE ON)
if(SANITIZER_AMDGPU)
add_definitions(-DSANITIZER_AMDGPU=1)
set(ASAN_LIB ${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}/lib/linux/libclang_rt.asan-x86_64.so)
endif()
target_link_libraries(hostrpc_services hsa-runtime64::hsa-runtime64 ${ASAN_LIB})
