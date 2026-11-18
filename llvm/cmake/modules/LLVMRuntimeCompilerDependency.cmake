include_guard(GLOBAL)

function(_llvm_track_compiler_dependency_dir dir)
  set(_compilers ${ARGN})
  if(NOT _compilers)
    return()
  endif()

  get_property(_targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
  foreach(_target IN LISTS _targets)
    get_target_property(_type ${_target} TYPE)
    if(NOT _type MATCHES "^(OBJECT_LIBRARY|STATIC_LIBRARY|SHARED_LIBRARY|MODULE_LIBRARY|EXECUTABLE)$")
      continue()
    endif()

    get_target_property(_sources ${_target} SOURCES)

    # Filter out generator expressions (e.g., $<TARGET_OBJECTS:...>).
    list(FILTER _sources EXCLUDE REGEX "^\\$<")
    if(NOT _sources)
      continue()
    endif()

    foreach(_compiler IN LISTS _compilers)
      set_property(SOURCE ${_sources}
        TARGET_DIRECTORY ${_target}
        APPEND PROPERTY OBJECT_DEPENDS "${_compiler}")
    endforeach()
  endforeach()

  # BUILDSYSTEM_TARGETS is flat, so recurse into subdirectories.
  get_property(_subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
  foreach(_subdir IN LISTS _subdirs)
    _llvm_track_compiler_dependency_dir("${_subdir}" ${_compilers})
  endforeach()
endfunction()

# cmake_language(DEFER CALL) re-evaluates arguments at call time in the
# directory scope.
function(_llvm_track_compiler_dependency_dir_deferred)
  get_property(_compilers DIRECTORY PROPERTY _LLVM_TRACK_COMPILERS)
  if(NOT _compilers)
    return()
  endif()
  _llvm_track_compiler_dependency_dir("${CMAKE_CURRENT_SOURCE_DIR}" ${_compilers})
endfunction()

# Add compiler binaries as target dependencies for code recompilation tracking.
#
# This function registers compiler binaries as build dependencies for all target
# objects compiled within the specified directory hierarchy. This guarantees
# that if a compiler binary is rebuilt, all runtime library objects are
# invalidated and rebuilt.
#
# This function must be called at the beginning of a runtime root CMakeLists.
#
# Argument is a ist of compiler binary file paths to track.
#
function(llvm_defer_compiler_dependency_tracking)
  set(COMPILERS ${ARGN})
  if(NOT COMPILERS)
    message(FATAL_ERROR "COMPILERS is required for llvm_defer_compiler_dependency_tracking")
  endif()

  # Store compilers as a directory property so the deferred call can read them.
  set_property(DIRECTORY APPEND PROPERTY _LLVM_TRACK_COMPILERS "${COMPILERS}")
  # Defer to triggers at the end of the current source dir.
  cmake_language(DEFER CALL _llvm_track_compiler_dependency_dir_deferred)
endfunction()
