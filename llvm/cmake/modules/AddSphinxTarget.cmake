
# Create sphinx target
if (LLVM_ENABLE_SPHINX AND NOT TARGET sphinx)
  message(STATUS "Sphinx enabled.")
  find_package(Sphinx REQUIRED)
  if (LLVM_BUILD_DOCS)
    add_custom_target(sphinx ALL)
  endif()
else()
  message(STATUS "Sphinx disabled.")
endif()


# Handy function for creating the different Sphinx targets.
#
# ``builder`` should be one of the supported builders used by
# the sphinx-build command.
#
# ``project`` should be the project name
function (add_sphinx_target builder project)
  set(SPHINX_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/${builder}")
  set(SPHINX_DOC_TREE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees-${builder}")
  set(SPHINX_TARGET_NAME docs-${project}-${builder})

  if (SPHINX_WARNINGS_AS_ERRORS)
    set(SPHINX_WARNINGS_AS_ERRORS_FLAG "-W")
  else()
    set(SPHINX_WARNINGS_AS_ERRORS_FLAG "")
  endif()

  add_custom_target(${SPHINX_TARGET_NAME}
                    COMMAND ${SPHINX_EXECUTABLE}
                            -b ${builder}
                            -d "${SPHINX_DOC_TREE_DIR}"
                            -q # Quiet: no output other than errors and warnings.
                            ${SPHINX_WARNINGS_AS_ERRORS_FLAG} # Treat warnings as errors if requested
                            "${CMAKE_CURRENT_SOURCE_DIR}" # Source
                            "${SPHINX_BUILD_DIR}" # Output
                    COMMENT
                    "Generating ${builder} Sphinx documentation for ${project} into \"${SPHINX_BUILD_DIR}\"")

  # When "clean" target is run, remove the Sphinx build directory
  set_property(DIRECTORY APPEND PROPERTY
               ADDITIONAL_MAKE_CLEAN_FILES
               "${SPHINX_BUILD_DIR}")

  # We need to remove ${SPHINX_DOC_TREE_DIR} when make clean is run
  # but we should only add this path once
  get_property(_CURRENT_MAKE_CLEAN_FILES
               DIRECTORY PROPERTY ADDITIONAL_MAKE_CLEAN_FILES)
  list(FIND _CURRENT_MAKE_CLEAN_FILES "${SPHINX_DOC_TREE_DIR}" _INDEX)
  if (_INDEX EQUAL -1)
    set_property(DIRECTORY APPEND PROPERTY
                 ADDITIONAL_MAKE_CLEAN_FILES
                 "${SPHINX_DOC_TREE_DIR}")
  endif()

  if (LLVM_BUILD_DOCS)
    add_dependencies(sphinx ${SPHINX_TARGET_NAME})

    # Handle installation
    if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
      if (builder STREQUAL man)
        if (CMAKE_INSTALL_MANDIR)
          set(INSTALL_MANDIR ${CMAKE_INSTALL_MANDIR}/)
        else()
          set(INSTALL_MANDIR share/man/)
        endif()
        # FIXME: We might not ship all the tools that these man pages describe
        install(DIRECTORY "${SPHINX_BUILD_DIR}/" # Slash indicates contents of
                COMPONENT "${project}-sphinx-man"
                DESTINATION ${INSTALL_MANDIR}man1)

      elseif (builder STREQUAL html)
        string(TOUPPER "${project}" project_upper)
        set(${project_upper}_INSTALL_SPHINX_HTML_DIR "share/doc/${project}/html"
            CACHE STRING "HTML documentation install directory for ${project}")

        # '/.' indicates: copy the contents of the directory directly into
        # the specified destination, without recreating the last component
        # of ${SPHINX_BUILD_DIR} implicitly.
        install(DIRECTORY "${SPHINX_BUILD_DIR}/."
                COMPONENT "${project}-sphinx-html"
                DESTINATION "${${project_upper}_INSTALL_SPHINX_HTML_DIR}")
      else()
        message(WARNING Installation of ${builder} not supported)
      endif()
    endif()
  endif()
endfunction()
