set(LLVM_BINUTILS_INCDIR "" CACHE PATH
  "PATH to binutils/include containing plugin-api.h for gold plugin.")

if( NOT LLVM_BINUTILS_INCDIR )
  # Nothing to say.
elseif( NOT EXISTS "${LLVM_BINUTILS_INCDIR}/plugin-api.h" )
  message(STATUS "plugin-api.h not found. gold plugin excluded from the build.")
else()
  include_directories( ${LLVM_BINUTILS_INCDIR} )

  # Because off_t is used in the public API, the largefile parts are required for
  # ABI compatibility.
  add_definitions( -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 )

  set(LLVM_LINK_COMPONENTS support)

  add_llvm_loadable_module(LLVMgold
    gold-plugin.cpp
    )

  # Makefile.rules contains a special cases for OpenBSD, Darwin and
  # Windows. We restrict ourselves to Linux for the time being.
  set(srcexp ${CMAKE_CURRENT_SOURCE_DIR}/gold.exports)
  add_custom_command(OUTPUT exportsfile
    COMMAND echo "{" > exportsfile
    COMMAND grep -q "\\<" ${srcexp} && echo "  global:" >> exportsfile || :
    COMMAND sed -e "s/$/;/" -e "s/^/    /" < ${srcexp} >> exportsfile
    COMMAND echo "  local: *;" >> exportsfile
    COMMAND echo "};" >> exportsfile
    DEPENDS ${srcexp}
    VERBATIM
    COMMENT "Creating export file for gold plugin")
  add_custom_target(gold_exports DEPENDS exportsfile)
  set_property(DIRECTORY APPEND
    PROPERTY ADDITIONAL_MAKE_CLEAN_FILES exportsfile)

  # Force re-linking when the exports file changes. Actually, it
  # forces recompilation of gold-plugin.cpp. The LINK_DEPENDS target
  # property only works for makefile-based generators.
  set_property(SOURCE gold-plugin.cpp APPEND PROPERTY
    OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/exportsfile)

  target_link_libraries(LLVMgold LTO
                   -Wl,--version-script,${CMAKE_CURRENT_BINARY_DIR}/exportsfile)
  add_dependencies(LLVMgold gold_exports)
endif()
