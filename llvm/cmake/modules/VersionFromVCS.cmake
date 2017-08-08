# Adds version control information to the variable VERS. For
# determining the Version Control System used (if any) it inspects the
# existence of certain subdirectories under SOURCE_DIR (if provided as an
# extra argument, otherwise uses CMAKE_CURRENT_SOURCE_DIR).

function(add_version_info_from_vcs VERS)
  SET(SOURCE_DIR ${ARGV1})
  if("${SOURCE_DIR}" STREQUAL "")
      SET(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  endif()
  string(REPLACE "svn" "" result "${${VERS}}")
  if( EXISTS "${SOURCE_DIR}/.svn" )
    set(result "${result}svn")
    # FindSubversion does not work with symlinks. See PR 8437
    if( NOT IS_SYMLINK "${SOURCE_DIR}" )
      find_package(Subversion)
    endif()
    if( Subversion_FOUND )
      subversion_wc_info( ${SOURCE_DIR} Project )
      if( Project_WC_REVISION )
        set(SVN_REVISION ${Project_WC_REVISION} PARENT_SCOPE)
        set(result "${result}-r${Project_WC_REVISION}")
      endif()
      if( Project_WC_URL )
        set(LLVM_REPOSITORY ${Project_WC_URL} PARENT_SCOPE)
      endif()
    endif()
  else()
    find_program(git_executable NAMES git git.exe git.cmd)

    if( git_executable )
      # Run from a subdirectory to force git to print an absoute path.
      execute_process(COMMAND ${git_executable} rev-parse --git-dir
        WORKING_DIRECTORY ${SOURCE_DIR}/cmake
        RESULT_VARIABLE git_result
        OUTPUT_VARIABLE git_dir
        ERROR_QUIET)
      if(git_result EQUAL 0)
        # Try to get a ref-id
        string(STRIP "${git_dir}" git_dir)
        set(result "${result}git")
        if( EXISTS ${git_dir}/svn )
          # Get the repository URL
          execute_process(COMMAND
            ${git_executable} svn info
            WORKING_DIRECTORY ${SOURCE_DIR}
            TIMEOUT 5
            RESULT_VARIABLE git_result
            OUTPUT_VARIABLE git_output
            ERROR_QUIET)
          if( git_result EQUAL 0 )
            string(REGEX MATCH "URL: ([^ \n]*)" svn_url ${git_output})
            if(svn_url)
              set(LLVM_REPOSITORY ${CMAKE_MATCH_1} PARENT_SCOPE)
            endif()
          endif()

          # Get the svn revision number for this git commit if one exists.
          execute_process(COMMAND ${git_executable} svn find-rev HEAD
            WORKING_DIRECTORY ${SOURCE_DIR}
            TIMEOUT 5
            RESULT_VARIABLE git_result
            OUTPUT_VARIABLE git_head_svn_rev_number
            OUTPUT_STRIP_TRAILING_WHITESPACE)
          if( git_result EQUAL 0 AND git_output)
            set(SVN_REVISION ${git_head_svn_rev_number} PARENT_SCOPE)
            set(git_svn_rev "-svn-${git_head_svn_rev_number}")
          else()
            set(git_svn_rev "")
          endif()
        endif()

        # Get the git ref id
        execute_process(COMMAND
          ${git_executable} rev-parse --short HEAD
          WORKING_DIRECTORY ${SOURCE_DIR}
          TIMEOUT 5
          RESULT_VARIABLE git_result
          OUTPUT_VARIABLE git_ref_id
          OUTPUT_STRIP_TRAILING_WHITESPACE)

        if( git_result EQUAL 0 )
          set(GIT_COMMIT ${git_ref_id} PARENT_SCOPE)
          set(result "${result}${git_svn_rev}-${git_ref_id}")
        else()
          set(result "${result}${git_svn_rev}")
        endif()
      endif()
    endif()
  endif()
  set(${VERS} ${result} PARENT_SCOPE)
endfunction(add_version_info_from_vcs)
