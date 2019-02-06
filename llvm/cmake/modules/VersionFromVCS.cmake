# Adds version control information to the variable VERS. For
# determining the Version Control System used (if any) it inspects the
# existence of certain subdirectories under SOURCE_DIR (if provided as an
# extra argument, otherwise uses CMAKE_CURRENT_SOURCE_DIR).

function(get_source_info_svn path revision repository)
  # If svn is a bat file, find_program(Subversion) doesn't find it.
  # Explicitly search for that here; Subversion_SVN_EXECUTABLE will override
  # the find_program call in FindSubversion.cmake.
  find_program(Subversion_SVN_EXECUTABLE NAMES svn svn.bat)
  find_package(Subversion)

  # Subversion module does not work with symlinks, see PR8437.
  get_filename_component(realpath ${path} REALPATH)
  if(Subversion_FOUND)
    subversion_wc_info(${realpath} Project)
    if(Project_WC_REVISION)
      set(${revision} ${Project_WC_REVISION} PARENT_SCOPE)
    endif()
    if(Project_WC_URL)
      set(${repository} ${Project_WC_URL} PARENT_SCOPE)
    endif()
  endif()
endfunction()

function(get_source_info_git path revision repository)
  find_package(Git)
  if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --git-dir
      WORKING_DIRECTORY ${path}
      RESULT_VARIABLE git_result
      OUTPUT_VARIABLE git_output
      ERROR_QUIET)
    if(git_result EQUAL 0)
      string(STRIP "${git_output}" git_output)
      get_filename_component(git_dir ${git_output} ABSOLUTE BASE_DIR ${path})
      if(EXISTS "${git_dir}/svn/refs")
        execute_process(COMMAND ${GIT_EXECUTABLE} svn info
          WORKING_DIRECTORY ${path}
          RESULT_VARIABLE git_result
          OUTPUT_VARIABLE git_output)
        if(git_result EQUAL 0)
          string(REGEX REPLACE "^(.*\n)?Revision: ([^\n]+).*"
            "\\2" git_svn_rev "${git_output}")
          set(${revision} ${git_svn_rev} PARENT_SCOPE)
          string(REGEX REPLACE "^(.*\n)?URL: ([^\n]+).*"
            "\\2" git_url "${git_output}")
          set(${repository} ${git_url} PARENT_SCOPE)
        endif()
      else()
        execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
          WORKING_DIRECTORY ${path}
          RESULT_VARIABLE git_result
          OUTPUT_VARIABLE git_output)
        if(git_result EQUAL 0)
          string(STRIP "${git_output}" git_output)
          set(${revision} ${git_output} PARENT_SCOPE)
        endif()
        execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref --symbolic-full-name @{upstream}
          WORKING_DIRECTORY ${path}
          RESULT_VARIABLE git_result
          OUTPUT_VARIABLE git_output
          ERROR_QUIET)
        if(git_result EQUAL 0)
          string(REPLACE "/" ";" branch ${git_output})
          list(GET branch 0 remote)
        else()
          set(remote "origin")
        endif()
        execute_process(COMMAND ${GIT_EXECUTABLE} remote get-url ${remote}
          WORKING_DIRECTORY ${path}
          RESULT_VARIABLE git_result
          OUTPUT_VARIABLE git_output
          ERROR_QUIET)
        if(git_result EQUAL 0)
          string(STRIP "${git_output}" git_output)
          set(${repository} ${git_output} PARENT_SCOPE)
        else()
          set(${repository} ${path} PARENT_SCOPE)
        endif()
      endif()
    endif()
  endif()
endfunction()

function(get_source_info path revision repository)
  if(EXISTS "${path}/.svn")
    get_source_info_svn("${path}" revision_info repository_info)
  else()
    get_source_info_git("${path}" revision_info repository_info)
  endif()
  set(${repository} "${repository_info}" PARENT_SCOPE)
  set(${revision} "${revision_info}" PARENT_SCOPE)
endfunction()
