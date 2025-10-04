# Example usage
# llvm_get_cache_vars(before)
# include(SomeModule)
# llvm_diff_cache_vars("${before}" new_vars new_pairs)

# message(STATUS "New cache variables: ${new_vars}")
# message(STATUS "New cache vars and values:\n${new_pairs}")

# get_list_of_existing_cache_variables(existing)
function(llvm_get_list_of_existing_cache_variables out_var)
  get_cmake_property(_all CACHE_VARIABLES)
  if(NOT _all)
    set(_all "")
  endif()
  set(${out_var} "${_all}" PARENT_SCOPE)
endfunction()

# list_of_new_cache_variables_and_values(existing new_vars_and_values)
# - `existing` is the name of the var returned by the first helper
# - `new_vars_and_values` will be a list like:  NAME=VALUE (TYPE=...);NAME2=VALUE2 (TYPE=...)
function(llvm_list_of_new_cache_variables_and_values existing_list_var out_var)
  # Existing (pre-include) snapshot
  set(_before "${${existing_list_var}}")

  # Current (post-include) snapshot
  get_cmake_property(_after CACHE_VARIABLES)

  # Compute new names
  set(_new "${_after}")
  if(_before)
    list(REMOVE_ITEM _new ${_before})
  endif()

  # Pack "NAME=VALUE (TYPE=...)" for each new cache entry
  set(_pairs "")
  foreach(_k IN LISTS _new)
    if(NOT "${_k}" MATCHES "^((C|CXX)_SUPPORTS|HAVE_|GLIBCXX_USE|SUPPORTS_FVISI)")
      continue()
    endif()
    # Cache VALUE: dereference is fine here because cache entries read like normal vars
    set(_val "${${_k}}")
    # Cache TYPE (e.g., STRING, BOOL, PATH, FILEPATH, INTERNAL, UNINITIALIZED)
    get_property(_type CACHE "${_k}" PROPERTY TYPE)
    if(NOT _type)
      set(_type "UNINITIALIZED")
    endif()
    list(APPEND _pairs "set(${_k} \"${_val}\" CACHE ${_type} \"\")")
  endforeach()

  set(${out_var} "${_pairs}" PARENT_SCOPE)
endfunction()
