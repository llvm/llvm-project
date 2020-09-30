set (CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/libompd/"
    ${CMAKE_MODULE_PATH}
)

find_package (Python3 COMPONENTS Interpreter Development)

execute_process(COMMAND "${Python3_EXECUTABLE}"
        "-mpip"
        "--version"
        OUTPUT_VARIABLE PIP_VERSION_INFO
        RESULT_VARIABLE HAD_ERROR)
string(REGEX REPLACE " " ";" PIP_VERSION_INFO "${PIP_VERSION_INFO}")
list(GET PIP_VERSION_INFO 1 PIP_VERSION)
set(PYSYSFLAG "")
if(PIP_VERSION VERSION_LESS_EQUAL "9.0.1" AND NOT DISABLE_SYSTEM_NON_DEBIAN)
  set(PYSYSFLAG "--system")
endif()

include_directories (${OMPD_INCLUDE_PATH})
include_directories (${LIBOMP_INCLUDE_DIR})

if(NOT PYTHON_HEADERS)
  set(PYTHON_HEADERS ${Python3_INCLUDE_DIRS})
endif()

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python-module/loadompd.py
                   DEPEND ompdModule.c loadompd.py setup.py ompd/frame_filter.py ompd/__init__.py ompd/ompd_address_space.py ompd/ompd_callbacks.py ompd/ompd_handles.py ompd/ompd.py
                   COMMAND ${CMAKE_COMMAND} -E env LIBOMP_INCLUDE_DIR=${LIBOMP_INCLUDE_DIR} env PYTHON_HEADERS=${PYTHON_HEADERS}
                   ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/setup.py -v bdist_wheel -b ${CMAKE_CURRENT_BINARY_DIR}/build -d ${CMAKE_CURRENT_BINARY_DIR}
                   COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/setup.py clean --all
                   COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/ompd.egg-info
                   COMMAND ${Python3_EXECUTABLE} -m pip install ${PYSYSFLAG} -U -t ${CMAKE_CURRENT_BINARY_DIR}/python-module --no-index --find-links=${CMAKE_CURRENT_BINARY_DIR} ompd
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(ompd_gdb_plugin ALL
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python-module/loadompd.py
                  COMMENT "Building the OMPD GDB plugin")

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python-module/ompd  DESTINATION ${OPENMP_INSTALL_LIBDIR})
