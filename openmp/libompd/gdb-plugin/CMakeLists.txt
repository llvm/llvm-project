#
#//===----------------------------------------------------------------------===//
#//
#// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#// See https://llvm.org/LICENSE.txt for license information.
#// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#//
#//===----------------------------------------------------------------------===//
#

set (CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/libompd/"
    ${CMAKE_MODULE_PATH}
)

find_package (Python3 COMPONENTS Interpreter Development)
find_package (PythonLibs REQUIRED)

include_directories (${OMPD_INCLUDE_PATH})
include_directories (${LIBOMP_INCLUDE_DIR})
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python-module/ompd/__init__.py
                   DEPENDS ompdModule.c ompdAPITests.c ompd/frame_filter.py ompd/__init__.py ompd/ompd_address_space.py ompd/ompd_callbacks.py ompd/ompd_handles.py ompd/ompd.py
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/ompd ${CMAKE_CURRENT_BINARY_DIR}/python-module/ompd/
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(ompd_gdb_plugin ALL
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python-module/ompd/__init__.py
                  COMMENT "Building the OMPD GDB plugin")

add_library (ompdModule MODULE ompdModule.c ompdAPITests.c)
include_directories (
        ${LIBOMP_INCLUDE_DIR}
        ${LIBOMP_SRC_DIR}
        ${Python3_INCLUDE_DIRS}
)
target_link_libraries (ompdModule ${Python3_LIBRARIES})
target_link_libraries (ompdModule ${CMAKE_DL_LIBS})

set_target_properties (ompdModule PROPERTIES PREFIX "")
set_target_properties (ompdModule PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python-module/ompd/")

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python-module/ompd DESTINATION ${CMAKE_INSTALL_PREFIX}/share/gdb/python/ PATTERN ompdModule.so PERMISSIONS OWNER_READ WORLD_READ GROUP_READ OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE)

