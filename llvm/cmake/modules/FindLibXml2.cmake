# Try to find the libxml2 library
#
# If successful, the following variables will be defined:
# LIBXML2_INCLUDE_DIR
# LIBXML2_LIBRARY
# LIBXML2_STATIC_LIBRARY
# LibXml2_FOUND
#
# Additionally, the following import targets will be defined:
# LibXml2::LibXml2
# LibXml2::LibXml2Static (if the static library is found)

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
  set(LIBXML2_LIBRARY "${LIBXML2_LIBRARIES}" CACHE FILEPATH "Path to libxml2 library")
endif()

find_library(LIBXML2_LIBRARY NAMES xml2 libxml2 libxml2s libxml2_a
  HINTS
  ${PC_LIBXML_LIBDIR}
  ${PC_LIBXML_LIBRARY_DIRS}
)

find_library(LIBXML2_STATIC_LIBRARY NAMES
  "${CMAKE_STATIC_LIBRARY_PREFIX}xml2${CMAKE_STATIC_LIBRARY_SUFFIX}"
  "${CMAKE_STATIC_LIBRARY_PREFIX}libxml2${CMAKE_STATIC_LIBRARY_SUFFIX}"
  HINTS
  ${PC_LIBXML_LIBDIR}
  ${PC_LIBXML_LIBRARY_DIRS}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibXml2
  REQUIRED_VARS LIBXML2_LIBRARY LIBXML2_INCLUDE_DIR
  VERSION_VAR PC_LIBXML_VERSION
)

if(LibXml2_FOUND)
  if(NOT TARGET LibXml2::LibXml2)
    add_library(LibXml2::LibXml2 UNKNOWN IMPORTED)
    set_target_properties(LibXml2::LibXml2 PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LIBXML2_INCLUDE_DIR}"
        IMPORTED_LOCATION "${LIBXML2_LIBRARY}")
  endif()
  if(LIBXML2_STATIC_LIBRARY AND NOT TARGET LibXml2::LibXml2Static)
    add_library(LibXml2::LibXml2Static STATIC IMPORTED)
    set_target_properties(LibXml2::LibXml2Static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LIBXML2_INCLUDE_DIR}"
        IMPORTED_LOCATION "${LIBXML2_STATIC_LIBRARY}")
    # Static libraries need their transitive dependencies for linking.
    set(LIBXML2_STATIC_DEPS)
    foreach(lib IN LISTS PC_LIBXML_STATIC_LIBRARIES)
      if(NOT lib STREQUAL "xml2")
        list(APPEND LIBXML2_STATIC_DEPS ${lib})
      endif()
    endforeach()
    if(LIBXML2_STATIC_DEPS)
      set_target_properties(LibXml2::LibXml2Static PROPERTIES
          INTERFACE_LINK_LIBRARIES "${LIBXML2_STATIC_DEPS}")
    endif()
  endif()
endif()

set(LIBXML2_INCLUDE_DIRS ${LIBXML2_INCLUDE_DIR})
set(LIBXML2_LIBRARIES ${LIBXML2_LIBRARY})
set(LIBXML2_DEFINITIONS ${PC_LIBXML_CFLAGS_OTHER})

mark_as_advanced(LIBXML2_INCLUDE_DIR LIBXML2_LIBRARY LIBXML2_STATIC_LIBRARY)
