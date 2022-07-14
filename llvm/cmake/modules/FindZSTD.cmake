find_path(ZSTD_INCLUDE_DIR
  NAMES zstd.h
  HINTS ${ZSTD_ROOT_DIR}/include)

find_library(ZSTD_LIBRARY
  NAMES zstd
  HINTS ${ZSTD_ROOT_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    ZSTD DEFAULT_MSG
    ZSTD_LIBRARY ZSTD_INCLUDE_DIR)

if(ZSTD_FOUND)
    set(ZSTD_LIBRARIES ${ZSTD_LIBRARY})
    set(ZSTD_INCLUDE_DIRS ${ZSTD_INCLUDE_DIR})
endif()

mark_as_advanced(
  ZSTD_INCLUDE_DIR
  ZSTD_LIBRARY)
