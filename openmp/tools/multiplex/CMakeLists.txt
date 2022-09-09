# Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
# Notified per clause 4(b) of the license.
if(LIBOMP_OMPT_SUPPORT)
  include_directories(${LIBOMP_INCLUDE_DIR})

  add_library(ompt-multiplex INTERFACE)
  target_include_directories(ompt-multiplex INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

  install(FILES ompt-multiplex.h DESTINATION "${LIBOMP_HEADERS_INSTALL_PATH}")

  add_subdirectory(tests)
endif()
