# Adds version control information to the variable VERS. For
# determining the Version Control System used (if any) it inspects the
# existence of certain subdirectories under CMAKE_CURRENT_SOURCE_DIR.

function(add_version_info_from_vcs VERS)
  string(REPLACE "svn" "" result "${${VERS}}")
  if( EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.svn" )
    set(result "${result}svn")
    # FindSubversion does not work with symlinks. See PR 8437
    if( NOT IS_SYMLINK "${CMAKE_CURRENT_SOURCE_DIR}" )
      find_package(Subversion)
    endif()
    if( Subversion_FOUND )
      subversion_wc_info( ${CMAKE_CURRENT_SOURCE_DIR} Project )
      if( Project_WC_REVISION )
        set(SVN_REVISION ${Project_WC_REVISION} PARENT_SCOPE)
        set(result "${result}-r${Project_WC_REVISION}")
      endif()
    endif()
  elseif( EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git )
    set(result "${result}git")
    # Try to get a ref-id
    find_program(git_executable NAMES git git.exe git.cmd)
    if( git_executable )
      set(is_git_svn_rev_exact false)
      execute_process(COMMAND ${git_executable} svn log --limit=1 --oneline
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                      TIMEOUT 5
                      RESULT_VARIABLE git_result
                      OUTPUT_VARIABLE git_output)
      if( git_result EQUAL 0 )
        string(REGEX MATCH r[0-9]+ git_svn_rev ${git_output})
        string(LENGTH "${git_svn_rev}" rev_length)
        math(EXPR rev_length "${rev_length}-1")
        string(SUBSTRING "${git_svn_rev}" 1 ${rev_length} git_svn_rev_number)
        set(SVN_REVISION ${git_svn_rev_number} PARENT_SCOPE)
        set(git_svn_rev "-svn-${git_svn_rev}")

        # Determine if the HEAD points directly at a subversion revision.
        execute_process(COMMAND ${git_executable} svn find-rev HEAD
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        TIMEOUT 5
                        RESULT_VARIABLE git_result
                        OUTPUT_VARIABLE git_output)
        if( git_result EQUAL 0 )
          string(STRIP "${git_output}" git_head_svn_rev_number)
          if( git_head_svn_rev_number EQUAL git_svn_rev_number )
            set(is_git_svn_rev_exact true)
          endif()
        endif()
      else()
        set(git_svn_rev "")
      endif()
      execute_process(COMMAND
                      ${git_executable} rev-parse --short HEAD
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                      TIMEOUT 5
                      RESULT_VARIABLE git_result
                      OUTPUT_VARIABLE git_output)
      if( git_result EQUAL 0 AND NOT is_git_svn_rev_exact )
        string(STRIP "${git_output}" git_ref_id)
        set(GIT_COMMIT ${git_ref_id} PARENT_SCOPE)
        set(result "${result}${git_svn_rev}-${git_ref_id}")
      else()
        set(result "${result}${git_svn_rev}")
      endif()
    endif()
  endif()
  set(${VERS} ${result} PARENT_SCOPE)
endfunction(add_version_info_from_vcs)
