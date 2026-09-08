include(GNUInstallDirs)

# Create sphinx target
if (LLVM_ENABLE_SPHINX)
  message(STATUS "Sphinx enabled.")
  find_package(Sphinx REQUIRED)

  # Sphinx has internal parallelism, so give it a custom job pool. Sphinx tends
  # not to use all available CPU, so this is not enabled by default.
  set(LLVM_PARALLEL_SPHINX_JOBS "" CACHE STRING
    "Define the maximum number of concurrent sphinx-build invocations (Ninja only).")
  if (LLVM_PARALLEL_SPHINX_JOBS)
    if (CMAKE_GENERATOR MATCHES "Ninja")
      get_property(_sphinx_job_pools GLOBAL PROPERTY JOB_POOLS)
      if (NOT "sphinx_job_pool=${LLVM_PARALLEL_SPHINX_JOBS}" IN_LIST _sphinx_job_pools)
        set_property(GLOBAL APPEND PROPERTY JOB_POOLS sphinx_job_pool=${LLVM_PARALLEL_SPHINX_JOBS})
      endif()
      set(sphinx_job_pool JOB_POOL sphinx_job_pool)
    else()
      message(WARNING "Job pooling is only available with Ninja generators.")
    endif()
  endif()

  if (LLVM_BUILD_DOCS AND NOT TARGET sphinx)
    add_custom_target(sphinx ALL)
    set_target_properties(sphinx PROPERTIES FOLDER "LLVM/Docs")
  endif()
else()
  message(STATUS "Sphinx disabled.")
endif()

# Create a target that synchronizes checked-in Sphinx inputs into a build-tree
# source directory. The synchronization preserves mtimes for unchanged files,
# updates changed files, and removes stale .rst/.md files so incremental docs
# builds stay correct when documentation is renamed or deleted. The cheap scan
# target is always run to notice directory listing changes, but it writes the
# file list only when it changes; the sync command depends on that list and on a
# depfile of the listed source files. The scan also records the destination
# .rst/.md listing so destination-only stale files trigger cleanup. It also
# records source files that are missing from the destination so externally
# removed copies are restored. The actual copy step can still be skipped when
# both trees are unchanged. Use PRESERVE_DOCS for generated .rst/.md files that
# live in the destination directory but do not exist in source_dir. Use
# IGNORE_MISSING_FILES for files that are copied by the sync step but
# intentionally removed by a later build action.
function(add_sphinx_source_sync_target target source_dir destination_dir)
  cmake_parse_arguments(ARG "" "" "PRESERVE_DOCS;IGNORE_MISSING_FILES" ${ARGN})

  set(sync_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${target}.sphinx-source-sync")
  set(file_list "${sync_dir}/files.txt")
  set(destination_doc_list "${sync_dir}/destination-docs.txt")
  set(missing_file_list "${sync_dir}/missing-files.txt")
  set(preserve_file "${sync_dir}/preserve.txt")
  set(ignore_missing_file "${sync_dir}/ignore-missing.txt")
  set(manifest_file "${sync_dir}/manifest.txt")
  set(depfile "${sync_dir}/sync.d")
  set(stamp_file "${sync_dir}/sync.stamp")
  set(scan_target "${target}-scan")
  set(scan_script "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/SphinxSourceScan.cmake")
  set(sync_script "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/SphinxSourceSync.cmake")
  set(utils_script "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/SphinxSourceUtils.cmake")

  file(MAKE_DIRECTORY "${sync_dir}")
  string(REPLACE ";" "\n" preserve_docs "${ARG_PRESERVE_DOCS}")
  if (preserve_docs)
    string(APPEND preserve_docs "\n")
  endif()
  if (EXISTS "${preserve_file}")
    file(READ "${preserve_file}" old_preserve_docs)
  else()
    set(old_preserve_docs)
  endif()
  if (NOT preserve_docs STREQUAL old_preserve_docs)
    file(WRITE "${preserve_file}" "${preserve_docs}")
  endif()
  string(REPLACE ";" "\n" ignore_missing_files "${ARG_IGNORE_MISSING_FILES}")
  if (ignore_missing_files)
    string(APPEND ignore_missing_files "\n")
  endif()
  if (EXISTS "${ignore_missing_file}")
    file(READ "${ignore_missing_file}" old_ignore_missing_files)
  else()
    set(old_ignore_missing_files)
  endif()
  if (NOT ignore_missing_files STREQUAL old_ignore_missing_files)
    file(WRITE "${ignore_missing_file}" "${ignore_missing_files}")
  endif()

  add_custom_target(${scan_target}
                    COMMAND "${CMAKE_COMMAND}"
                            "-DSOURCE_DIR=${source_dir}"
                            "-DDESTINATION_DIR=${destination_dir}"
                            "-DFILE_LIST=${file_list}"
                            "-DDESTINATION_DOC_LIST=${destination_doc_list}"
                            "-DMISSING_FILE_LIST=${missing_file_list}"
                            "-DIGNORE_MISSING_FILE=${ignore_missing_file}"
                            -P "${scan_script}"
                    BYPRODUCTS "${file_list}" "${destination_doc_list}"
                               "${missing_file_list}"
                    DEPENDS "${scan_script}" "${utils_script}"
                    COMMENT
                    "Scanning Sphinx sources in \"${source_dir}\""
                    VERBATIM)

  add_custom_command(OUTPUT "${stamp_file}"
                     COMMAND "${CMAKE_COMMAND}"
                             "-DSOURCE_DIR=${source_dir}"
                             "-DDESTINATION_DIR=${destination_dir}"
                             "-DFILE_LIST=${file_list}"
                             "-DPRESERVE_FILE=${preserve_file}"
                             "-DMANIFEST_FILE=${manifest_file}"
                             "-DDEPFILE=${depfile}"
                             "-DSTAMP_FILE=${stamp_file}"
                             -P "${sync_script}"
                     DEPENDS "${file_list}" "${preserve_file}"
                             "${destination_doc_list}" "${missing_file_list}"
                             "${ignore_missing_file}" "${sync_script}"
                             "${utils_script}"
                     DEPFILE "${depfile}"
                     BYPRODUCTS "${manifest_file}"
                     COMMENT
                     "Copying Sphinx sources from \"${source_dir}\" to \"${destination_dir}\""
                     VERBATIM)

  add_custom_target(${target} DEPENDS "${stamp_file}")
  add_dependencies(${target} ${scan_target})
endfunction()

# Handy function for creating the different Sphinx targets.
#
# ``builder`` should be one of the supported builders used by
# the sphinx-build command.
#
# ``project`` should be the project name
#
# Named arguments:
# ``ENV_VARS`` should be a list of environment variables that should be set when
#              running Sphinx. Each environment variable should be a string with
#              the form KEY=VALUE.
function (add_sphinx_target builder project)
  cmake_parse_arguments(ARG "" "SOURCE_DIR" "ENV_VARS" ${ARGN})
  set(SPHINX_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/${builder}")
  set(SPHINX_DOC_TREE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees-${project}-${builder}")
  set(SPHINX_TARGET_NAME docs-${project}-${builder})

  if (SPHINX_WARNINGS_AS_ERRORS)
    set(SPHINX_WARNINGS_AS_ERRORS_FLAG "-W")
  else()
    set(SPHINX_WARNINGS_AS_ERRORS_FLAG "")
  endif()

  if (NOT ARG_SOURCE_DIR)
    set(ARG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()

  # Give Sphinx some internal job parallelism, since it tends to be on the
  # critical path at the end of the build. This can speed up doc builds by
  # ~80%. Sphinx rarely consumes all cores available, so it's safe to
  # overallocate a bit.
  if (NOT DEFINED LLVM_SPHINX_THREADS)
    cmake_host_system_information(RESULT number_of_logical_cores
                                  QUERY NUMBER_OF_LOGICAL_CORES)
    math(EXPR LLVM_SPHINX_THREADS
         "(${number_of_logical_cores} + 1) / 2")
  endif()
  set(LLVM_SPHINX_THREADS "${LLVM_SPHINX_THREADS}"
      CACHE STRING "Define the number of parallel jobs for each Sphinx build.")
  if (LLVM_SPHINX_THREADS)
    set(sphinx_jobs_flag -j ${LLVM_SPHINX_THREADS})
  endif()

  if ("${LLVM_VERSION_SUFFIX}" STREQUAL "git")
    set(PreReleaseTag "-tPreRelease")
  endif()

  add_custom_target(${SPHINX_TARGET_NAME}
                    COMMAND ${CMAKE_COMMAND} -E env ${ARG_ENV_VARS}
                            --modify "PYTHONPATH=path_list_append:${LLVM_MAIN_SRC_DIR}/../utils/docs"
                            ${SPHINX_EXECUTABLE}
                            -b ${builder}
                            -d "${SPHINX_DOC_TREE_DIR}"
                            ${sphinx_jobs_flag}
                            -q # Quiet: no output other than errors and warnings.
                            -t builder-${builder} # tag for builder
                            -D version=${LLVM_VERSION_MAJOR}
                            -D release=${PACKAGE_VERSION}
                            ${PreReleaseTag}
                            ${SPHINX_WARNINGS_AS_ERRORS_FLAG} # Treat warnings as errors if requested
                            "${ARG_SOURCE_DIR}" # Source
                            "${SPHINX_BUILD_DIR}" # Output
                    ${sphinx_job_pool}
                    COMMENT
                    "Generating ${builder} Sphinx documentation for ${project} into \"${SPHINX_BUILD_DIR}\"")
  get_subproject_title(subproject_title)
  set_target_properties(${SPHINX_TARGET_NAME} PROPERTIES FOLDER "${subproject_title}/Docs")

  # When "clean" target is run, remove the Sphinx build directory
  set_property(DIRECTORY APPEND PROPERTY
               ADDITIONAL_MAKE_CLEAN_FILES
               "${SPHINX_BUILD_DIR}")

  # We need to remove ${SPHINX_DOC_TREE_DIR} when make clean is run
  # but we should only add this path once
  get_property(_CURRENT_MAKE_CLEAN_FILES
               DIRECTORY PROPERTY ADDITIONAL_MAKE_CLEAN_FILES)
  if (NOT "${SPHINX_DOC_TREE_DIR}" IN_LIST _CURRENT_MAKE_CLEAN_FILES)
    set_property(DIRECTORY APPEND PROPERTY
                 ADDITIONAL_MAKE_CLEAN_FILES
                 "${SPHINX_DOC_TREE_DIR}")
  endif()

  if (LLVM_BUILD_DOCS)
    add_dependencies(sphinx ${SPHINX_TARGET_NAME})

    # Handle installation
    if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
      if (builder STREQUAL man)
        # FIXME: We might not ship all the tools that these man pages describe
        install(DIRECTORY "${SPHINX_BUILD_DIR}/" # Slash indicates contents of
                COMPONENT "${project}-sphinx-man"
                DESTINATION "${CMAKE_INSTALL_MANDIR}/man1")

        if(NOT LLVM_ENABLE_IDE)
          add_llvm_install_targets("install-${SPHINX_TARGET_NAME}"
                                   DEPENDS ${SPHINX_TARGET_NAME}
                                   COMPONENT "${project}-sphinx-man")
        endif()
      elseif (builder STREQUAL html)
        string(TOUPPER "${project}" project_upper)
        set(${project_upper}_INSTALL_SPHINX_HTML_DIR "${CMAKE_INSTALL_DOCDIR}/${project}/html"
            CACHE STRING "HTML documentation install directory for ${project}")

        # '/.' indicates: copy the contents of the directory directly into
        # the specified destination, without recreating the last component
        # of ${SPHINX_BUILD_DIR} implicitly.
        install(DIRECTORY "${SPHINX_BUILD_DIR}/."
                COMPONENT "${project}-sphinx-html"
                DESTINATION "${${project_upper}_INSTALL_SPHINX_HTML_DIR}")

        if(NOT LLVM_ENABLE_IDE)
          add_llvm_install_targets("install-${SPHINX_TARGET_NAME}"
                                   DEPENDS ${SPHINX_TARGET_NAME}
                                   COMPONENT "${project}-sphinx-html")
        endif()
      else()
        message(WARNING Installation of ${builder} not supported)
      endif()
    endif()
  endif()
endfunction()
