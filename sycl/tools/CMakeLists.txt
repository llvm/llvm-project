set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(get_device_count_by_type get_device_count_by_type.cpp)
add_dependencies(get_device_count_by_type ocl-headers ocl-icd)
#Minimum supported version of Intel's OCL GPU and CPU devices
add_definitions(-D MIN_INTEL_OCL_GPU_VERSION=\\"18.47.11882\\")
add_definitions(-D MIN_INTEL_OCL_CPU_VERSION=\\"18.1.0.0901\\",\\"7.6.0.1202\\")

add_executable(sycl-check sycl-check.cpp)
add_dependencies(sycl-check sycl)
target_link_libraries(sycl-check sycl)
