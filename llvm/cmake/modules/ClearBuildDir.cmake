# CMake script that clears a project build directory
# while preserving CMake file API queries.
#
# Input variables:
#   BINARY_DIR  - The build directory to remove.

set(API_QUERY_DIR "${BINARY_DIR}/.cmake/api/v1/query")

if(IS_DIRECTORY "${API_QUERY_DIR}")
  # BINARY_DIR usually ends with a slash, strip it to make
  # the query backup directory name.
  string(REGEX REPLACE "/+$" "" BINARY_DIR_BASE "${BINARY_DIR}")
  set(QUERY_BACKUP "${BINARY_DIR_BASE}.api_query_backup")
  file(RENAME "${API_QUERY_DIR}" "${QUERY_BACKUP}")
endif()

file(REMOVE_RECURSE "${BINARY_DIR}")

if(DEFINED QUERY_BACKUP)
  file(MAKE_DIRECTORY "${BINARY_DIR}/.cmake/api/v1")
  file(RENAME "${QUERY_BACKUP}" "${API_QUERY_DIR}")
endif()
