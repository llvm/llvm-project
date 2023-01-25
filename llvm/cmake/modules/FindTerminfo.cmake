# Attempts to discover terminfo library with a linkable setupterm function.
#
# Example usage:
#
# find_package(Terminfo)
#
# If successful, the following variables will be defined:
# Terminfo_FOUND
# Terminfo_LIBRARIES
#
# Additionally, the following import target will be defined:
# Terminfo::terminfo

find_library(Terminfo_LIBRARIES NAMES terminfo tinfo curses ncurses ncursesw)

if(Terminfo_LIBRARIES)
  include(CMakePushCheckState)
  include(CheckCSourceCompiles)
  cmake_push_check_state()
  list(APPEND CMAKE_REQUIRED_LIBRARIES ${Terminfo_LIBRARIES})
  check_c_source_compiles("
    int setupterm(char *term, int filedes, int *errret);
    int main(void) { return setupterm(0, 0, 0); }"
    Terminfo_LINKABLE)
  cmake_pop_check_state()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Terminfo
                                  FOUND_VAR
                                    Terminfo_FOUND
                                  REQUIRED_VARS
                                    Terminfo_LIBRARIES
                                    Terminfo_LINKABLE)
mark_as_advanced(Terminfo_LIBRARIES
                 Terminfo_LINKABLE)

if(Terminfo_FOUND)
  if(NOT TARGET Terminfo::terminfo)
    add_library(Terminfo::terminfo UNKNOWN IMPORTED)
    set_target_properties(Terminfo::terminfo PROPERTIES IMPORTED_LOCATION "${Terminfo_LIBRARIES}")
  endif()
endif()
