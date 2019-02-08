# Looking for Z3 in LLVM_Z3_INSTALL_DIR
find_path(Z3_INCLUDE_DIR NAMES z3.h
   NO_DEFAULT_PATH
   PATHS ${LLVM_Z3_INSTALL_DIR}/include
   PATH_SUFFIXES libz3 z3
   )

find_library(Z3_LIBRARIES NAMES z3 libz3
   NO_DEFAULT_PATH
   PATHS ${LLVM_Z3_INSTALL_DIR}
   PATH_SUFFIXES lib bin
   )

find_program(Z3_EXECUTABLE z3
   NO_DEFAULT_PATH
   PATHS ${LLVM_Z3_INSTALL_DIR}
   PATH_SUFFIXES bin
   )

# If Z3 has not been found in LLVM_Z3_INSTALL_DIR look in the default directories
find_path(Z3_INCLUDE_DIR NAMES z3.h
   PATH_SUFFIXES libz3 z3
   )

find_library(Z3_LIBRARIES NAMES z3 libz3
   PATH_SUFFIXES lib bin
   )

find_program(Z3_EXECUTABLE z3
   PATH_SUFFIXES bin
   )

if(Z3_INCLUDE_DIR AND Z3_LIBRARIES AND Z3_EXECUTABLE)
    execute_process (COMMAND ${Z3_EXECUTABLE} -version
      OUTPUT_VARIABLE libz3_version_str
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    string(REGEX REPLACE "^Z3 version ([0-9.]+)" "\\1"
           Z3_VERSION_STRING "${libz3_version_str}")
    unset(libz3_version_str)
endif()

# handle the QUIETLY and REQUIRED arguments and set Z3_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Z3
                                  REQUIRED_VARS Z3_LIBRARIES Z3_INCLUDE_DIR
                                  VERSION_VAR Z3_VERSION_STRING)

mark_as_advanced(Z3_INCLUDE_DIR Z3_LIBRARIES)
