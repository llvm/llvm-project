# CMake project that writes Subversion revision information to a header.
#
# Input variables:
#   SOURCE_DIRS - A list of source directories.
#   NAMES       - A list of macro prefixes for each of the source directories.
#   HEADER_FILE - The header file to write
#
# The output header will contain macros <NAME>_REPOSITORY and <NAME>_REVISION,
# where "<NAME>" and is substituted with the names specified in the input
# variables, for each of the SOURCE_DIRS given.

# Chop off cmake/modules/GetSVN.cmake
get_filename_component(LLVM_DIR "${CMAKE_SCRIPT_MODE_FILE}" PATH)
get_filename_component(LLVM_DIR "${LLVM_DIR}" PATH)
get_filename_component(LLVM_DIR "${LLVM_DIR}" PATH)

# Handle strange terminals
set(ENV{TERM} "dumb")

macro(get_source_info_svn path revision repository)
  # If svn is a bat file, find_program(Subversion) doesn't find it.
  # Explicitly search for that here; Subversion_SVN_EXECUTABLE will override
  # the find_program call in FindSubversion.cmake.
  find_program(Subversion_SVN_EXECUTABLE NAMES svn svn.bat)

  # FindSubversion does not work with symlinks. See PR 8437
  if (NOT IS_SYMLINK "${path}")
    find_package(Subversion)
  endif()
  if (Subversion_FOUND)
    subversion_wc_info( ${path} Project )
    if (Project_WC_REVISION)
      set(${revision} ${Project_WC_REVISION} PARENT_SCOPE)
    endif()
    if (Project_WC_URL)
      set(${repository} ${Project_WC_URL} PARENT_SCOPE)
    endif()
  endif()
endmacro()

macro(get_source_info_git_svn path revision repository)
  find_program(git_executable NAMES git git.exe git.cmd)
  if (git_executable)
    execute_process(COMMAND ${git_executable} svn info
      WORKING_DIRECTORY ${path}
      TIMEOUT 5
      RESULT_VARIABLE git_result
      OUTPUT_VARIABLE git_output)
    if (git_result EQUAL 0)
      string(REGEX REPLACE "^(.*\n)?Revision: ([^\n]+).*"
        "\\2" git_svn_rev "${git_output}")
      set(${revision} ${git_svn_rev} PARENT_SCOPE)
      string(REGEX REPLACE "^(.*\n)?URL: ([^\n]+).*"
        "\\2" git_url "${git_output}")
      set(${repository} ${git_url} PARENT_SCOPE)
    endif()
  endif()
endmacro()

macro(get_source_info_git path revision repository)
  find_program(git_executable NAMES git git.exe git.cmd)
  if (git_executable)
    execute_process(COMMAND ${git_executable} log -1 --pretty=format:%H
      WORKING_DIRECTORY ${path}
      TIMEOUT 5
      RESULT_VARIABLE git_result
      OUTPUT_VARIABLE git_output)
    if (git_result EQUAL 0)
      set(${revision} ${git_output} PARENT_SCOPE)
    endif()
    execute_process(COMMAND ${git_executable} remote -v
      WORKING_DIRECTORY ${path}
      TIMEOUT 5
      RESULT_VARIABLE git_result
      OUTPUT_VARIABLE git_output)
    if (git_result EQUAL 0)
      string(REGEX REPLACE "^(.*\n)?[^ \t]+[ \t]+([^ \t\n]+)[ \t]+\\(fetch\\).*"
        "\\2" git_url "${git_output}")
      set(${repository} "${git_url}" PARENT_SCOPE)
    endif()
  endif()
endmacro()

function(get_source_info path revision repository)
  if (EXISTS "${path}/.svn")
    get_source_info_svn("${path}" revision repository)
  elseif (EXISTS "${path}/.git/svn/refs")
    get_source_info_git_svn("${path}" revision repository)
  elseif (EXISTS "${path}/.git")
    get_source_info_git("${path}" revision repository)
  endif()
endfunction()

function(append_info name path)
  get_source_info("${path}" revision repository)
  string(STRIP "${revision}" revision)
  string(STRIP "${repository}" repository)
  file(APPEND "${HEADER_FILE}.txt"
    "#define ${name}_REVISION \"${revision}\"\n")
  file(APPEND "${HEADER_FILE}.txt"
    "#define ${name}_REPOSITORY \"${repository}\"\n")
endfunction()

function(validate_inputs source_dirs names)
  list(LENGTH source_dirs source_dirs_length)
  list(LENGTH names names_length)
  if (NOT source_dirs_length EQUAL names_length)
    message(FATAL_ERROR
            "GetSVN.cmake takes two arguments: a list of source directories, "
            "and a list of names. Expected two lists must be of equal length, "
            "but got ${source_dirs_length} source directories and "
            "${names_length} names.")
  endif()
endfunction()

if (DEFINED SOURCE_DIRS AND DEFINED NAMES)
  validate_inputs("${SOURCE_DIRS}" "${NAMES}")

  list(LENGTH SOURCE_DIRS source_dirs_length)
  math(EXPR source_dirs_max_index ${source_dirs_length}-1)
  foreach(index RANGE ${source_dirs_max_index})
    list(GET SOURCE_DIRS ${index} source_dir)
    list(GET NAMES ${index} name)
    append_info(${name} ${source_dir})
  endforeach()
endif()

# Allow -DFIRST_SOURCE_DIR arguments until Clang migrates to the new
# -DSOURCE_DIRS argument.
if(DEFINED FIRST_SOURCE_DIR)
  append_info(${FIRST_NAME} "${FIRST_SOURCE_DIR}")
  if(DEFINED SECOND_SOURCE_DIR)
    append_info(${SECOND_NAME} "${SECOND_SOURCE_DIR}")
  endif()
endif()

# Copy the file only if it has changed.
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${HEADER_FILE}.txt" "${HEADER_FILE}")
file(REMOVE "${HEADER_FILE}.txt")

