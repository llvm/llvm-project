# Returns library name for a given path.
function(get_library_name path name)
  get_filename_component(path ${path} NAME)
  set(prefixes ${CMAKE_FIND_LIBRARY_PREFIXES})
  set(suffixes ${CMAKE_FIND_LIBRARY_SUFFIXES})
  list(FILTER prefixes EXCLUDE REGEX "^\\s*$")
  list(FILTER suffixes EXCLUDE REGEX "^\\s*$")
  # Do not strip the "lib" prefix for Windows because MSVC-style linkers don't
  # implicitly add the "lib" prefix.
  if(prefixes AND NOT Win32)
    string(REPLACE ";" "|" prefixes "${prefixes}")
    string(REGEX REPLACE "^(${prefixes})" "" path ${path})
  endif()
  if(suffixes)
    string(REPLACE ";" "|" suffixes "${suffixes}")
    string(REGEX REPLACE "(${suffixes})$" "" path ${path})
  endif()
  set(${name} "${path}" PARENT_SCOPE)
endfunction()
