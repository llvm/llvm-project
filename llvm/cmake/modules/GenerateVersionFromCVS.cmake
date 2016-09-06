# CMake project that writes Subversion revision information to a header.
#
# Input variables:
#   SRC               - Source directory
#   HEADER_FILE       - The header file to write
#
# The output header will contain macros FIRST_REPOSITORY and FIRST_REVISION,
# and SECOND_REPOSITORY and SECOND_REVISION if requested, where "FIRST" and
# "SECOND" are substituted with the names specified in the input variables.



# Chop off cmake/modules/GetSVN.cmake
get_filename_component(LLVM_DIR "${CMAKE_SCRIPT_MODE_FILE}" PATH)
get_filename_component(LLVM_DIR "${LLVM_DIR}" PATH)
get_filename_component(LLVM_DIR "${LLVM_DIR}" PATH)

set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${LLVM_DIR}/cmake/modules")
include(VersionFromVCS)

# Handle strange terminals
set(ENV{TERM} "dumb")

function(append_info name path)
  add_version_info_from_vcs(REVISION ${path})
  string(STRIP "${REVISION}" REVISION)
  file(APPEND "${HEADER_FILE}.txt"
    "#define ${name} \"${REVISION}\"\n")
endfunction()

append_info(${NAME} "${SOURCE_DIR}")

# Copy the file only if it has changed.
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${HEADER_FILE}.txt" "${HEADER_FILE}")
file(REMOVE "${HEADER_FILE}.txt")

