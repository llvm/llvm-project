# Handy function to export symbols without versioning
function(add_symbol_exports target_name export_file)
  if(LLVM_HAVE_LINK_VERSION_SCRIPT)
    # Gold and BFD ld require a version script rather than a plain list.
    set(native_export_file "${target_name}.exports")
    add_custom_command(
      OUTPUT ${native_export_file}
      COMMAND echo "{" > ${native_export_file}
      COMMAND grep -q "[[:alnum:]]" ${export_file} && echo "  global:" >>
              ${native_export_file} || :
      COMMAND sed -e "s/$/;/" -e "s/^/    /" < ${export_file} >>
              ${native_export_file}
      COMMAND echo "  local: *;" >> ${native_export_file}
      COMMAND echo "};" >> ${native_export_file}
      DEPENDS ${export_file}
      VERBATIM
      COMMENT "Creating export file for ${target_name}")
    set_property(
    TARGET ${target_name}
    APPEND_STRING
    PROPERTY
        LINK_FLAGS
        "  -Wl,--version-script,\"${CMAKE_CURRENT_BINARY_DIR}/${native_export_file}\""
    )
  endif()

  add_custom_target(${target_name}_exports DEPENDS ${native_export_file})
  set_target_properties(${target_name}_exports PROPERTIES FOLDER "Misc")

  get_property(
    srcs
    TARGET ${target_name}
    PROPERTY SOURCES)
  foreach(src ${srcs})
    get_filename_component(extension ${src} EXT)
    if(extension STREQUAL ".cpp")
      set(first_source_file ${src})
      break()
    endif()
  endforeach()

  # Force re-linking when the exports file changes. Actually, it forces
  # recompilation of the source file. The LINK_DEPENDS target property only
  # works for makefile-based generators. FIXME: This is not safe because this
  # will create the same target ${native_export_file} in several different file:
  # - One where we emitted ${target_name}_exports - One where we emitted the
  # build command for the following object. set_property(SOURCE
  # ${first_source_file} APPEND PROPERTY OBJECT_DEPENDS
  # ${CMAKE_CURRENT_BINARY_DIR}/${native_export_file})

  set_property(
    DIRECTORY
    APPEND
    PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${native_export_file})

  add_dependencies(${target_name} ${target_name}_exports)

  # Add dependency to *_exports later -- CMake issue 14747
  list(APPEND LLVM_COMMON_DEPENDS ${target_name}_exports)
  set(LLVM_COMMON_DEPENDS
      ${LLVM_COMMON_DEPENDS}
      PARENT_SCOPE)
endfunction(add_symbol_exports)
