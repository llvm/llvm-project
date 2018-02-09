find_first_existing_vc_file(llvm_vc "${LLVM_MAIN_SRC_DIR}")

# The VC revision include that we want to generate.
set(version_inc "${CMAKE_CURRENT_BINARY_DIR}/VCSRevision.h")

set(get_svn_script "${LLVM_CMAKE_PATH}/GenerateVersionFromCVS.cmake")

file(WRITE "${version_inc}.undef" "#undef LLVM_REVISION\n")
if((DEFINED llvm_vc) AND LLVM_APPEND_VC_REV)

  execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files
      "${version_inc}.undef" "${version_inc}"
      RESULT_VARIABLE files_not_equal
      OUTPUT_QUIET
      ERROR_QUIET)
  # Remove ${version_inc} if it doesn't define a revision. This will force it
  # to be regenerated when toggling LLVM_APPEND_VC_REV from OFF to ON.
  if(NOT files_not_equal)
    file(REMOVE "${version_inc}")
  endif()

  # Create custom target to generate the VC revision include.
  add_custom_command(OUTPUT "${version_inc}"
    DEPENDS "${llvm_vc}" "${get_svn_script}"
    COMMAND
    ${CMAKE_COMMAND} "-DSOURCE_DIR=${LLVM_MAIN_SRC_DIR}"
                     "-DNAME=LLVM_REVISION"
                     "-DHEADER_FILE=${version_inc}"
                     -P "${get_svn_script}")
else()
  # Make sure ${version_inc} doesn't define a revision
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${version_inc}.undef" "${version_inc}")
endif()
file(REMOVE "${version_inc}.undef")

# Mark the generated header as being generated.
set_source_files_properties("${version_inc}"
  PROPERTIES GENERATED TRUE
             HEADER_FILE_ONLY TRUE)

add_custom_target(llvm_vcsrevision_h DEPENDS "${version_inc}")
set_target_properties(llvm_vcsrevision_h PROPERTIES FOLDER "Misc")
