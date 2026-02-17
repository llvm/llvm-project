add_lldb_library(lldbTreeSitter
  TreeSitterHighlighter.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbUtility
    ${TreeSitter_LIBRARY}
)

function(add_tree_sitter_grammar name source_dir binary_dir)
  set(TreeSitter_SOURCE_DIR ${source_dir}/${name})
  set(TreeSitter_BINARY_DIR ${binary_dir}/src)

  add_custom_command(
    OUTPUT  ${TreeSitter_BINARY_DIR}/parser.c
    COMMAND ${TreeSitter_BINARY} generate -o ${TreeSitter_BINARY_DIR}
    DEPENDS ${TreeSitter_SOURCE_DIR}/grammar.js
    WORKING_DIRECTORY ${TreeSitter_SOURCE_DIR}
    VERBATIM
  )

  add_lldb_library(${name}
    ${TreeSitter_BINARY_DIR}/parser.c
    ${TreeSitter_SOURCE_DIR}/scanner.c
  )
  set_property(TARGET ${name} PROPERTY C_STANDARD 11)
  target_compile_options(${name} PRIVATE
      "-Wno-everything"
  )
  target_include_directories(${name} PRIVATE
    ${binary_dir}/src
  )

  file(READ ${TreeSitter_SOURCE_DIR}/highlights.scm HIGHLIGHT_QUERY)
  configure_file(
      "${source_dir}/HighlightQuery.h.in"
      "${binary_dir}/HighlightQuery.h"
      @ONLY
  )
endfunction()

add_subdirectory(Swift)
