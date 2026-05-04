# CheckSize.cmake
# Verifies that the combined EJIT archive fits within the size budget.
#
# Input variables (passed via -D):
#   ARCHIVE_FILE — path to the archive to check
#   MAX_SIZE_MB  — size budget in MB (default 10)

cmake_minimum_required(VERSION 3.20)

if(NOT EXISTS "${ARCHIVE_FILE}")
  message(FATAL_ERROR "Archive not found: ${ARCHIVE_FILE}")
endif()

if(NOT MAX_SIZE_MB)
  set(MAX_SIZE_MB 10)
endif()

file(SIZE "${ARCHIVE_FILE}" _bytes)
math(EXPR _max_bytes "${MAX_SIZE_MB} * 1048576")
math(EXPR _mb "${_bytes} / 1048576")
math(EXPR _kb "(${_bytes} % 1048576) / 1024")

message(STATUS "=============================================")
message(STATUS "  EJIT Minimal Archive Size Check")
message(STATUS "=============================================")
message(STATUS "  File:       ${ARCHIVE_FILE}")
message(STATUS "  Size:       ${_bytes} bytes (${_mb}.${_kb} MB)")
message(STATUS "  Budget:     ${MAX_SIZE_MB} MB")
message(STATUS "---------------------------------------------")

if(_bytes LESS_EQUAL _max_bytes)
  math(EXPR _pct "${_bytes} * 100 / ${_max_bytes}")
  message(STATUS "  Result:     PASS (${_pct}% of budget)")
  message(STATUS "=============================================")
else()
  math(EXPR _over "${_bytes} - ${_max_bytes}")
  math(EXPR _over_kb "${_over} / 1024")
  math(EXPR _pct "${_bytes} * 100 / ${_max_bytes}")
  message(STATUS "  Result:     OVER BUDGET by ${_over_kb} KB (${_pct}% of budget)")
  message(STATUS "=============================================")
  message(FATAL_ERROR "Archive exceeds ${MAX_SIZE_MB} MB budget")
endif()
