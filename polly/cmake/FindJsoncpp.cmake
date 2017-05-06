find_package(PkgConfig QUIET)
if (PkgConfig_FOUND)
  pkg_search_module(JSONCPP jsoncpp QUIET)

  # Get the libraries full paths, to be consistent with find_library().
  set(fulllibs)
  foreach (libname IN LISTS JSONCPP_LIBRARIES)
    find_library(lib NAMES ${libname} 
      HINTS ${JSONCPP_LIBDIR} ${JSONCPP_LIBRARY_DIRS}
      NO_DEFAULT_PATH
    )
    if (lib)
      list(APPEND fulllibs ${lib})
    else ()
      list(APPEND fulllibs ${libname})
    endif ()
  endforeach ()
  set(JSONCPP_LIBRARIES ${fulllibs})

  set(JSONCPP_DEFINITIONS ${JSONCPP_CFLAGS})
else ()
  set(JSONCPP_DEFINITIONS)

  find_path(JSONCPP_INCLUDE_DIR json/json.h
    PATHS ENV JSONCPP_INCLUDE ENV JSONCPP_DIR
    PATH_SUFFIXES jsoncpp
    NO_DEFAULT_PATH
  )
  find_path(JSONCPP_INCLUDE_DIR json/json.h
    PATH_SUFFIXES jsoncpp
  )
  mark_as_advanced(JSONCPP_INCLUDE_DIR)
  set(JSONCPP_INCLUDE_DIRS "${JSONCPP_INCLUDE_DIR}")

  find_library(JSONCPP_LIBRARY NAMES jsoncpp
    HINTS ENV JSONCPP_LIB ENV JSONCPP_DIR
    NO_DEFAULT_PATH
  )
  find_library(JSONCPP_LIBRARY NAMES jsoncpp)
  mark_as_advanced(JSONCPP_LIBRARY)
  set(JSON_LIBRARIES ${JSON_LIBRARY})
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Jsoncpp DEFAULT_MSG JSONCPP_INCLUDE_DIRS JSONCPP_LIBRARIES JSONCPP_DEFINITIONS)

if (Jsoncpp_FOUND)
  add_library(jsoncpp INTERFACE IMPORTED)
  foreach (incl IN LISTS JSONCPP_INCLUDE_DIRS)
    set_property(TARGET jsoncpp APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${incl})
  endforeach ()
  foreach (libname IN LISTS JSONCPP_LIBRARIES)
    set_property(TARGET jsoncpp APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${lib})
  endforeach ()
  foreach (opt IN LISTS JSONCPP_DEFINITIONS)
    set_property(TARGET jsoncpp APPEND PROPERTY INTERFACE_COMPILE_OPTIONS ${opt})
  endforeach ()
endif ()
