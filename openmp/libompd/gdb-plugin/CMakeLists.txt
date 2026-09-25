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

include_directories (${OMPD_INCLUDE_PATH})
include_directories (${LIBOMP_INCLUDE_DIR})
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python-module/ompd/__init__.py
                   DEPENDS ompdModule.c ompdDLService.cpp ompdAPITests.c ompd/frame_filter.py ompd/__init__.py ompd/ompd_address_space.py ompd/ompd_callbacks.py ompd/ompd_handles.py ompd/ompd.py
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/ompd ${CMAKE_CURRENT_BINARY_DIR}/python-module/ompd/
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(ompd_gdb_plugin ALL
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python-module/ompd/__init__.py
                  COMMENT "Building the OMPD GDB plugin")

add_library (ompdModule MODULE ompdModule.c ompdAPITests.c ompdDLService.cpp)
include_directories (
        ${LIBOMP_INCLUDE_DIR}
        ${LIBOMP_SRC_DIR}
        ${Python3_INCLUDE_DIRS}
)
if(LLVM_INCLUDE_DIRS)
    target_include_directories(ompdModule PRIVATE ${LLVM_INCLUDE_DIRS})
else()
    target_include_directories(ompdModule PRIVATE ${LLVM_INCLUDE_DIR} ${LLVM_MAIN_INCLUDE_DIR})
endif()
target_link_libraries (ompdModule ${Python3_LIBRARIES})
# Use LLVM Support's DynamicLibrary, not POSIX dlopen, and not libclang-cpp.
if(LLVM_LINK_LLVM_DYLIB)
    target_link_libraries(ompdModule LLVM)
else()
    llvm_map_components_to_libnames(ompd_llvm_libs Support)
    target_link_libraries(ompdModule ${ompd_llvm_libs})
endif()

set_target_properties (ompdModule PROPERTIES PREFIX "")
set_target_properties (ompdModule PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties (ompdModule PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python-module/ompd/")

# ompdModule links LLVMSupport or libLLVM. GDB loads it from
# ${CMAKE_INSTALL_DATADIR}/gdb/python/ompd/, not from the LLVM library dir. Set
# RUNPATH so the dynamic linker finds those LLVM libs without LD_LIBRARY_PATH.
# install(DIRECTORY) copies files as-is and does not rewrite RPATH, so the
# .so must be installed with install(TARGETS).
set(_ompd_gdb_plugin_dest "${CMAKE_INSTALL_DATADIR}/gdb/python")
if(UNIX AND NOT APPLE)
  if(LLVM_LIBRARY_DIR)
    set_property(TARGET ompdModule APPEND PROPERTY BUILD_RPATH "${LLVM_LIBRARY_DIR}")
  endif()
  set(_ompd_mod_install_dir "${CMAKE_INSTALL_FULL_DATADIR}/gdb/python/ompd")
  set(_ompd_llvm_lib_install "${CMAKE_INSTALL_FULL_LIBDIR}")
  file(RELATIVE_PATH _ompd_llvm_lib_rel
       "${_ompd_mod_install_dir}" "${_ompd_llvm_lib_install}")
  set_target_properties(ompdModule PROPERTIES
                        INSTALL_RPATH "\$ORIGIN;\$ORIGIN/${_ompd_llvm_lib_rel}")
endif()

openmp_register_meta_dep(ompd_gdb_plugin ompdModule)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python-module/ompd
        DESTINATION ${_ompd_gdb_plugin_dest}
        COMPONENT openmp
        PATTERN ompdModule.so EXCLUDE)
install(TARGETS ompdModule
        LIBRARY DESTINATION ${_ompd_gdb_plugin_dest}/ompd
        COMPONENT openmp)

