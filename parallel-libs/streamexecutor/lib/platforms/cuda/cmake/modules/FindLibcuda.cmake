# - Try to find the libcuda library
# Once done this will define
#  LIBCUDA_FOUND - System has libcuda
#  LIBCUDA_INCLUDE_DIRS - The libcuda include directories
#  LIBCUDA_LIBRARIES - The libraries needed to use libcuda

# TODO(jhen): Allow users to specify a search path.
find_path(LIBCUDA_INCLUDE_DIR cuda.h /usr/local/cuda/include)
# TODO(jhen): Use the library that goes with the headers.
find_library(LIBCUDA_LIBRARY cuda)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBCUDA_FOUND to TRUE if
# all listed variables are TRUE
find_package_handle_standard_args(
    LIBCUDA DEFAULT_MSG LIBCUDA_INCLUDE_DIR LIBCUDA_LIBRARY)

mark_as_advanced(LIBCUDA_INCLUDE_DIR LIBCUDA_LIBRARY)

set(LIBCUDA_LIBRARIES ${LIBCUDA_LIBRARY})
set(LIBCUDA_INCLUDE_DIRS ${LIBCUDA_INCLUDE_DIR})
