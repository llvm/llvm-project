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

file(READ "/etc/os-release" OS_RELEASE)
set(DIST "")
string(REGEX MATCH "Debian|Ubuntu" DIST ${OS_RELEASE})

# UBUNTU and Debian package manager have its own patch in "pip" to avoid user
# installed packages messing up with default paths.
# https://bugs.launchpad.net/ubuntu/+source/python-pip/+bug/1419695
# Therfore, we have to use "--system" (specific to ubuntu and debian) when we 
# use system installed pip.(Not required if user installed pip on other paths.)
# However, this has been taken care in pip for 20+ versions.
# https://github.com/pypa/pip/commit/5f1468274987348b569aa586eeca4363494d0357

if(DIST)
    execute_process(COMMAND "${Python3_EXECUTABLE}"
            "-mpip"
            "--version"
            OUTPUT_VARIABLE PIP_VERSION_INFO
            RESULT_VARIABLE HAD_ERROR)
    if (NOT ${HAD_ERROR} EQUAL 0)
      message(WARNING "PIP command failed, gdb-plugin disabled.")
      return()
    endif ()
    string(REGEX REPLACE " " ";" PIP_VERSION_INFO "${PIP_VERSION_INFO}")
    list(GET PIP_VERSION_INFO 1 PIP_VERSION)
    set(PYSYSFLAG "")

    if(PIP_VERSION VERSION_LESS "20.0.0")
      execute_process(COMMAND "${Python3_EXECUTABLE}"
                "-mpip"
                "install"
                "--help"
                OUTPUT_VARIABLE PIP_INSTALL_HELP
                RESULT_VARIABLE HAD_ERROR )
      string(REGEX MATCH "--system" SYSTEM_FLAG ${PIP_INSTALL_HELP})
      if (SYSTEM_FLAG)
        set(PYSYSFLAG "--system")
      endif()
    endif()
endif()

include_directories (${OMPD_INCLUDE_PATH})
include_directories (${LIBOMP_INCLUDE_DIR})
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python-module/ompd/__init__.py
                   DEPENDS ompdModule.c ompdAPITests.c setup.py ompd/frame_filter.py ompd/__init__.py ompd/ompd_address_space.py ompd/ompd_callbacks.py ompd/ompd_handles.py ompd/ompd.py
                   COMMAND ${CMAKE_COMMAND} -E env LIBOMP_INCLUDE_DIR=${LIBOMP_INCLUDE_DIR}
                   ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/setup.py -v bdist_wheel -b ${CMAKE_CURRENT_BINARY_DIR}/build -d ${CMAKE_CURRENT_BINARY_DIR}
                   COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/setup.py clean --all
                   COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/ompd.egg-info
                   COMMAND ${Python3_EXECUTABLE} -m pip install ${PYSYSFLAG} -U -t ${CMAKE_CURRENT_BINARY_DIR}/python-module --no-index --find-links=${CMAKE_CURRENT_BINARY_DIR} ompd
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(ompd_gdb_plugin ALL
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python-module/ompd/__init__.py
                  COMMENT "Building the OMPD GDB plugin")

install(CODE "execute_process(COMMAND ${Python3_EXECUTABLE} -m pip install ${PYSYSFLAG} -U -t ${CMAKE_INSTALL_PREFIX}/share/gdb/python/gdb --no-index --find-links=${CMAKE_CURRENT_BINARY_DIR} ompd)")

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python-module/ompd  DESTINATION "${DEVEL_PACKAGE}${OPENMP_INSTALL_LIBDIR}")
