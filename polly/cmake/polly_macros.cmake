
include(CMakeParseArguments)

macro(add_polly_library name)
  cmake_parse_arguments(ARG "" "" "" ${ARGN})
  set(srcs ${ARG_UNPARSED_ARGUMENTS})
  if(MSVC_IDE OR XCODE)
    file( GLOB_RECURSE headers *.h *.td *.def)
    set(srcs ${srcs} ${headers})
    string( REGEX MATCHALL "/[^/]+" split_path ${CMAKE_CURRENT_SOURCE_DIR})
    list( GET split_path -1 dir)
    file( GLOB_RECURSE headers
      ../../include/polly${dir}/*.h)
    set(srcs ${srcs} ${headers})
  endif(MSVC_IDE OR XCODE)
  if (MODULE)
    set(libkind MODULE)
  elseif (SHARED_LIBRARY)
    set(libkind SHARED)
  else()
    set(libkind)
  endif()
  add_library( ${name} ${libkind} ${srcs} )
  set_target_properties(${name} PROPERTIES FOLDER "Polly")

  if( LLVM_COMMON_DEPENDS )
    add_dependencies( ${name} ${LLVM_COMMON_DEPENDS} )
  endif( LLVM_COMMON_DEPENDS )
  if( LLVM_USED_LIBS )
    foreach(lib ${LLVM_USED_LIBS})
      target_link_libraries( ${name} PUBLIC ${lib} )
    endforeach(lib)
  endif( LLVM_USED_LIBS )

  if(POLLY_LINK_LIBS)
    foreach(lib ${POLLY_LINK_LIBS})
      target_link_libraries(${name} PUBLIC ${lib})
    endforeach(lib)
  endif(POLLY_LINK_LIBS)

  if( LLVM_LINK_COMPONENTS )
    llvm_config(${name} ${LLVM_LINK_COMPONENTS})
  endif( LLVM_LINK_COMPONENTS )
  if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY OR ${name} STREQUAL "LLVMPolly")
    install(TARGETS ${name}
      COMPONENT ${name}
      EXPORT LLVMExports
      LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX})
    add_llvm_install_targets(install-${name}
      COMPONENT ${name})
  endif()
  set_property(GLOBAL APPEND PROPERTY LLVM_EXPORTS ${name})
endmacro(add_polly_library)

macro(add_polly_loadable_module name)
  set(srcs ${ARGN})
  # klduge: pass different values for MODULE with multiple targets in same dir
  # this allows building shared-lib and module in same dir
  # there must be a cleaner way to achieve this....
  if (MODULE)
  else()
    set(GLOBAL_NOT_MODULE TRUE)
  endif()
  set(MODULE TRUE)
  add_polly_library(${name} ${srcs})
  set_target_properties(${name} PROPERTIES FOLDER "Polly")
  if (GLOBAL_NOT_MODULE)
    unset (MODULE)
  endif()
  if (APPLE)
    # Darwin-specific linker flags for loadable modules.
    set_target_properties(${name} PROPERTIES
      LINK_FLAGS "-Wl,-flat_namespace -Wl,-undefined -Wl,suppress")
  endif()
endmacro(add_polly_loadable_module)

# Recursive helper for setup_source_group. Traverse the file system and add
# source files matching the glob_expr to the prefix, recursing into
# subdirectories as they are encountered
function(setup_polly_source_groups_helper pwd prefix glob_expr)
  file(GLOB children RELATIVE ${pwd} ${pwd}/*)
  foreach(child ${children})
    if (IS_DIRECTORY ${pwd}/${child})
      setup_polly_source_groups_helper(${pwd}/${child}
        "${prefix}\\${child}" ${glob_expr})
    endif()
  endforeach()

  file(GLOB to_add ${pwd}/${glob_expr})
  source_group(${prefix} FILES ${to_add})
endfunction(setup_polly_source_groups_helper)

# Set up source groups in order to nicely organize source files in IDEs
macro(setup_polly_source_groups src_root hdr_root)
  # FIXME: The helper can be eliminated if the CMake version is increased
  # to 3.8 or higher. If this is done, the TREE version of source_group can
  # be used
  setup_polly_source_groups_helper(${src_root} "Source Files" "*.cpp")
  setup_polly_source_groups_helper(${hdr_root} "Header Files" "*.h")
endmacro(setup_polly_source_groups)
