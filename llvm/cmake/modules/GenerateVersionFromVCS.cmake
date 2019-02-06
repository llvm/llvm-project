# CMake script that writes version control information to a header.
#
# Input variables:
#   NAMES             - A list of names for each of the source directories.
#   <NAME>_SOURCE_DIR - A path to source directory for each name in NAMES.
#   HEADER_FILE       - The header file to write
#
# The output header will contain macros <NAME>_REPOSITORY and <NAME>_REVISION,
# where "<NAME>" is substituted with the names specified in the input variables,
# for each of the <NAME>_SOURCE_DIR given.

get_filename_component(LLVM_DIR "${CMAKE_SCRIPT_MODE_FILE}" PATH)
get_filename_component(LLVM_DIR "${LLVM_DIR}" PATH)
get_filename_component(LLVM_DIR "${LLVM_DIR}" PATH)

list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}/cmake/modules")

include(VersionFromVCS)

# Handle strange terminals
set(ENV{TERM} "dumb")

function(append_info name path)
  if(path)
    get_source_info("${path}" revision repository)
  endif()
  if(revision)
    file(APPEND "${HEADER_FILE}.tmp"
      "#define ${name}_REVISION \"${revision}\"\n")
  else()
    file(APPEND "${HEADER_FILE}.tmp"
      "#undef ${name}_REVISION\n")
  endif()
  if(repository)
    file(APPEND "${HEADER_FILE}.tmp"
      "#define ${name}_REPOSITORY \"${repository}\"\n")
  else()
    file(APPEND "${HEADER_FILE}.tmp"
      "#undef ${name}_REPOSITORY\n")
  endif()
endfunction()

foreach(name IN LISTS NAMES)
  if(NOT DEFINED ${name}_SOURCE_DIR)
    message(FATAL_ERROR "${name}_SOURCE_DIR is not defined")
  endif()
  append_info(${name} "${${name}_SOURCE_DIR}")
endforeach()

# Copy the file only if it has changed.
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${HEADER_FILE}.tmp" "${HEADER_FILE}")
file(REMOVE "${HEADER_FILE}.tmp")
