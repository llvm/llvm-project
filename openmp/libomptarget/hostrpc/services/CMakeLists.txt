
find_package(hsa-runtime64 1.2.0 REQUIRED HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)
add_library(hostrpc_services STATIC hostcall.cpp hostrpc_execute_service.c hostrpc_externs.c)
set_property(TARGET hostrpc_services PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(hostrpc_services hsa-runtime64::hsa-runtime64)
