# Based on https://gitlab.kitware.com/cmake/cmake/-/blob/3986f4b79ea6bf247eefad7ddb883cd6f65ac5c1/Modules/FindLibXml2.cmake
# With support for using a static libxml2 library

# use pkg-config to get the directories and then use these values
# in the find_path() and find_library() calls
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PC_LIBXML QUIET libxml-2.0)
endif()

find_path(LIBXML2_INCLUDE_DIR NAMES libxml/xpath.h
  HINTS
  ${PC_LIBXML_INCLUDEDIR}
  ${PC_LIBXML_INCLUDE_DIRS}
  PATH_SUFFIXES libxml2
)

if(DEFINED LIBXML2_LIBRARIES AND NOT DEFINED LIBXML2_LIBRARY)
  set(LIBXML2_LIBRARY ${LIBXML2_LIBRARIES})
endif()

find_library(LIBXML2_SHARED_LIBRARY NAMES xml2 libxml2 libxml2_a
  HINTS
  ${PC_LIBXML_LIBDIR}
  ${PC_LIBXML_LIBRARY_DIRS}
)

# This is a system lib on macOS, so we don't need to avoid the dependency
if(APPLE)
  set(LLVM_USE_STATIC_LIBXML2 OFF)
endif()

if(LLVM_USE_STATIC_LIBXML2)
  set(_original_suffixes ${CMAKE_FIND_LIBRARY_SUFFIXES})
  if(UNIX)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
  elseif(WIN32)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a)
  endif()

  find_library(LIBXML2_STATIC_LIBRARY NAMES xml2 libxml2 libxml2_a
    HINTS
    ${PC_LIBXML_LIBDIR}
    ${PC_LIBXML_LIBRARY_DIRS}
  )

  if(LIBXML2_STATIC_LIBRARY STREQUAL "LIBXML2_STATIC_LIBRARY-NOTFOUND")
    message(FATAL_ERROR "Static libxml2 requested (LLVM_USE_STATIC_LIBXML2=ON) but not found")
  endif()

  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_original_suffixes})
  set(LIBXML2_LIBRARY ${LIBXML2_STATIC_LIBRARY})
else()
  set(LIBXML2_LIBRARY ${LIBXML2_SHARED_LIBRARY})
endif()

set(LIBXML2_LIBRARIES ${LIBXML2_LIBRARY})
set(LIBXML2_INCLUDE_DIRS ${LIBXML2_INCLUDE_DIR})

unset(LIBXML2_DEFINITIONS)
foreach(libxml2_pc_lib_dir IN LISTS PC_LIBXML_LIBDIR PC_LIBXML_LIBRARY_DIRS)
  if (LIBXML2_LIBRARY MATCHES "^${libxml2_pc_lib_dir}")
    list(APPEND LIBXML2_INCLUDE_DIRS ${PC_LIBXML_INCLUDE_DIRS})
    set(LIBXML2_DEFINITIONS ${PC_LIBXML_CFLAGS_OTHER})
    break()
  endif()
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibXml2
                                  REQUIRED_VARS LIBXML2_LIBRARY LIBXML2_INCLUDE_DIR
                                  VERSION_VAR LibXml2_VERSION)

mark_as_advanced(LIBXML2_INCLUDE_DIR LIBXML2_LIBRARY)

if(LibXml2_FOUND AND NOT TARGET LibXml2::LibXml2)
  add_library(LibXml2::LibXml2 UNKNOWN IMPORTED)
  set_target_properties(LibXml2::LibXml2 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LIBXML2_INCLUDE_DIRS}")
  set_target_properties(LibXml2::LibXml2 PROPERTIES INTERFACE_COMPILE_OPTIONS "${LIBXML2_DEFINITIONS}")
  set_property(TARGET LibXml2::LibXml2 APPEND PROPERTY IMPORTED_LOCATION "${LIBXML2_LIBRARY}")

  add_library(LibXml2::LibXml2Shared UNKNOWN IMPORTED)
  set_target_properties(LibXml2::LibXml2Shared PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LIBXML2_INCLUDE_DIRS}")
  set_target_properties(LibXml2::LibXml2Shared PROPERTIES INTERFACE_COMPILE_OPTIONS "${LIBXML2_DEFINITIONS}")
  set_property(TARGET LibXml2::LibXml2Shared APPEND PROPERTY IMPORTED_LOCATION "${LIBXML2_SHARED_LIBRARY}")
endif()
