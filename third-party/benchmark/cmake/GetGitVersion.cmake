# - Returns a version string from Git tags
#
# This function inspects the annotated git tags for the project and returns a string
# into a CMake variable
#
#  get_git_version(<var>)
#
# - Example
#
# include(GetGitVersion)
# get_git_version(GIT_VERSION)
#
# Requires CMake 2.8.11+
find_package(Git)

if(__get_git_version)
  return()
endif()
set(__get_git_version INCLUDED)

function(get_git_version var)
  if(GIT_EXECUTABLE)
      execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --match "v[0-9]*.[0-9]*.[0-9]*" --abbrev=8
          WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
          RESULT_VARIABLE status
          OUTPUT_VARIABLE GIT_DESCRIBE_VERSION
          ERROR_QUIET)
      if(status)
          set(GIT_DESCRIBE_VERSION "v0.0.0")
      endif()
      
      string(STRIP ${GIT_DESCRIBE_VERSION} GIT_DESCRIBE_VERSION)
      if(GIT_DESCRIBE_VERSION MATCHES v[^-]*-) 
         string(REGEX REPLACE "v([^-]*)-([0-9]+)-.*" "\\1.\\2"  GIT_VERSION ${GIT_DESCRIBE_VERSION})
      else()
         string(REGEX REPLACE "v(.*)" "\\1" GIT_VERSION ${GIT_DESCRIBE_VERSION})
      endif()

      # Work out if the repository is dirty
      execute_process(COMMAND ${GIT_EXECUTABLE} update-index -q --refresh
          WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
          OUTPUT_QUIET
          ERROR_QUIET)
      execute_process(COMMAND ${GIT_EXECUTABLE} diff-index --name-only HEAD --
          WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
          OUTPUT_VARIABLE GIT_DIFF_INDEX
          ERROR_QUIET)
      string(COMPARE NOTEQUAL "${GIT_DIFF_INDEX}" "" GIT_DIRTY)
      if (${GIT_DIRTY})
          set(GIT_DESCRIBE_VERSION "${GIT_DESCRIBE_VERSION}-dirty")
      endif()
      message(STATUS "git version: ${GIT_DESCRIBE_VERSION} normalized to ${GIT_VERSION}")
  else()
      set(GIT_VERSION "0.0.0")
  endif()

  set(${var} ${GIT_VERSION} PARENT_SCOPE)
endfunction()
