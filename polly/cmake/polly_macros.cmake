
include(CMakeParseArguments)

macro(add_polly_library name)
  cmake_parse_arguments(ARG "FORCE_STATIC" "" "" ${ARGN})
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
  elseif (ARG_FORCE_STATIC)
    if (SHARED_LIBRARY OR BUILD_SHARED_LIBS)
      message(STATUS "${name} is being built as static library because it is compiled with -fvisibility=hidden; "
                     "Its symbols are not visible from outside a shared library")
    endif ()
    set(libkind STATIC)
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
      target_link_libraries( ${name} ${lib} )
    endforeach(lib)
  endif( LLVM_USED_LIBS )

  if(POLLY_LINK_LIBS)
    foreach(lib ${POLLY_LINK_LIBS})
      target_link_libraries(${name} ${lib})
    endforeach(lib)
  endif(POLLY_LINK_LIBS)

  if( LLVM_LINK_COMPONENTS )
    llvm_config(${name} ${LLVM_LINK_COMPONENTS})
  endif( LLVM_LINK_COMPONENTS )
  if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY OR ${name} STREQUAL "LLVMPolly")
    install(TARGETS ${name}
      EXPORT LLVMExports
      LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX})
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

# Use C99-compatible compile mode for all C source files of a target.
function(target_enable_c99 _target)
  if(CMAKE_VERSION VERSION_GREATER "3.1")
    set_target_properties("${_target}" PROPERTIES C_STANDARD 99)
  elseif(CMAKE_COMPILER_IS_GNUCC)
    get_target_property(_sources "${_target}" SOURCES)
    foreach(_file IN LISTS _sources)
      get_source_file_property(_lang "${_file}" LANGUAGE)
      if(_lang STREQUAL "C")
        set_source_files_properties(${_file} COMPILE_FLAGS "-std=gnu99")
      endif()
    endforeach()
  endif()
endfunction()
