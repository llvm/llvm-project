# CombineArchives.cmake
# Extracts objects from all listed static libraries and repackages them
# into a single libejit_minimal.a for embedded deployment.
#
# Input variables (passed via -D):
#   LIB_LIST_FILE — path to a text file listing .a files to combine
#   OUTPUT        — path for the output combined archive
#   CMAKE_AR      — path to ar/llvm-ar

cmake_minimum_required(VERSION 3.20)

if(NOT EXISTS "${LIB_LIST_FILE}")
  message(FATAL_ERROR "Library list not found: ${LIB_LIST_FILE}")
endif()

file(READ "${LIB_LIST_FILE}" _libs_raw)
string(STRIP "${_libs_raw}" _libs_raw)
string(REPLACE "\n" ";" _libs "${_libs_raw}")

set(TMPDIR "${CMAKE_CURRENT_BINARY_DIR}/ejit_obj_tmp")
file(REMOVE_RECURSE "${TMPDIR}")
file(MAKE_DIRECTORY "${TMPDIR}")

set(_total_objects 0)
set(_missing 0)

foreach(_lib_path ${_libs})
  if(NOT EXISTS "${_lib_path}")
    message(STATUS "  [SKIP] missing: ${_lib_path}")
    math(EXPR _missing "${_missing} + 1")
    continue()
  endif()

  string(REGEX REPLACE ".*/lib(.*)\\.a$" "\\1" _name "${_lib_path}")
  execute_process(
    COMMAND ${CMAKE_AR} x "${_lib_path}"
    WORKING_DIRECTORY "${TMPDIR}"
    RESULT_VARIABLE _res
    ERROR_QUIET
    )
  if(_res EQUAL 0)
    file(GLOB _objs "${TMPDIR}/*.o")
    list(LENGTH _objs _count)
    message(STATUS "  Extracted ${_name}: ${_count} objects")
    math(EXPR _total_objects "${_total_objects} + ${_count}")
  endif()
endforeach()

file(GLOB _all_objs "${TMPDIR}/*.o")
list(LENGTH _all_objs _final_count)

if(_final_count EQUAL 0)
  file(REMOVE_RECURSE "${TMPDIR}")
  message(FATAL_ERROR "No objects extracted. Check that libraries are built.")
endif()

execute_process(
  COMMAND ${CMAKE_AR} crs "${OUTPUT}" ${_all_objs}
  WORKING_DIRECTORY "${TMPDIR}"
  RESULT_VARIABLE _res
  )

file(REMOVE_RECURSE "${TMPDIR}")

if(NOT _res EQUAL 0)
  message(FATAL_ERROR "Failed to create ${OUTPUT}")
endif()

# Report file size
file(SIZE "${OUTPUT}" _bytes)
math(EXPR _kb "${_bytes} / 1024")
math(EXPR _mb "${_bytes} / 1048576")
message(STATUS "Created: ${OUTPUT}")
message(STATUS "  Objects: ${_final_count}  |  Size: ${_bytes} bytes (${_kb} KB / ${_mb} MB)")
if(_missing GREATER 0)
  message(WARNING "  ${_missing} libraries were not found and were skipped")
endif()
