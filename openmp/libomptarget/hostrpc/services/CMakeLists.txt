
if (OPENMP_ENABLE_LIBOMPTARGET_HSA)
  add_library(hostrpc_services STATIC hostcall.cpp hostrpc_execute_service.c hostrpc_externs.c)
  set_property(TARGET hostrpc_services PROPERTY POSITION_INDEPENDENT_CODE ON)
  find_path(HSA_INCLUDE hsa.h HINTS ${CMAKE_INSTALL_PREFIX}/include/hsa ${CMAKE_INSTALL_PREFIX}/../include/hsa PATH_SUFFIXES hsa)
  include_directories(${HSA_INCLUDE})
endif()
