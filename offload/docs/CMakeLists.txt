if(LLVM_ENABLE_SPHINX)
  include(AddSphinxTarget)
  if(SPHINX_FOUND AND SPHINX_OUTPUT_HTML)
    # Generate offload-api.rst from OffloadAPI.td
    set(LLVM_TARGET_DEFINITIONS
      ${OFFLOAD_SOURCE_DIR}/liboffload/API/OffloadAPI.td)
    tablegen(OFFLOAD source/offload-api.rst -gen-doc
      EXTRA_INCLUDES ${OFFLOAD_SOURCE_DIR}/liboffload/API)
    add_public_tablegen_target(OffloadDocsGenerate)

    # Due to Sphinx only allowing a single source direcotry and the fact we
    # only generate a single file, copy offload-api.rst to the source directory
    # to be included in the generated documentation.
    # Additionally, copy the llvm-theme into the Sphinx source directory.
    # A .gitignore file ensures the copied files will not be added to the
    # repository.
    add_custom_target(OffloadDocsCopy
      COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_BINARY_DIR}/source/offload-api.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/offload-api.rst
      COMMAND ${CMAKE_COMMAND} -E copy
        ${OFFLOAD_SOURCE_DIR}/../clang/www/favicon.ico
        ${CMAKE_CURRENT_SOURCE_DIR}/_static/favicon.ico
      COMMAND ${CMAKE_COMMAND} -E copy
        ${OFFLOAD_SOURCE_DIR}/../llvm/docs/_static/llvm.css
        ${CMAKE_CURRENT_SOURCE_DIR}/_static/llvm.css
      COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${OFFLOAD_SOURCE_DIR}/../llvm/docs/_themes
        ${CMAKE_CURRENT_SOURCE_DIR}/_themes
    )
    add_dependencies(OffloadDocsCopy OffloadDocsGenerate)

    # Generate the HTML documentation, the docs-offload-html target.
    add_sphinx_target(html offload)
    add_dependencies(docs-offload-html OffloadDocsCopy)
  endif()
endif()
