function(get_system_libs return_var)
  message(AUTHOR_WARNING "get_system_libs no longer needed")
  set(${return_var} "" PARENT_SCOPE)
endfunction()


function(link_system_libs target)
  message(AUTHOR_WARNING "link_system_libs no longer needed")
endfunction()


function(is_llvm_target_library library return_var)
  # Sets variable `return_var' to ON if `library' corresponds to a
  # LLVM supported target. To OFF if it doesn't.
  set(${return_var} OFF PARENT_SCOPE)
  string(TOUPPER "${library}" capitalized_lib)
  string(TOUPPER "${LLVM_ALL_TARGETS}" targets)
  foreach(t ${targets})
    if( capitalized_lib STREQUAL t OR
        capitalized_lib STREQUAL "LLVM${t}" OR
        capitalized_lib STREQUAL "LLVM${t}CODEGEN" OR
        capitalized_lib STREQUAL "LLVM${t}ASMPARSER" OR
        capitalized_lib STREQUAL "LLVM${t}ASMPRINTER" OR
        capitalized_lib STREQUAL "LLVM${t}DISASSEMBLER" OR
        capitalized_lib STREQUAL "LLVM${t}INFO" )
      set(${return_var} ON PARENT_SCOPE)
      break()
    endif()
  endforeach()
endfunction(is_llvm_target_library)


macro(llvm_config executable)
  explicit_llvm_config(${executable} ${ARGN})
endmacro(llvm_config)


function(explicit_llvm_config executable)
  set( link_components ${ARGN} )

  llvm_map_components_to_libnames(LIBRARIES ${link_components})
  get_target_property(t ${executable} TYPE)
  if("${t}" STREQUAL "STATIC_LIBRARY")
    target_link_libraries(${executable} ${cmake_2_8_12_INTERFACE} ${LIBRARIES})
  elseif("${t}" STREQUAL "SHARED_LIBRARY" OR "${t}" STREQUAL "MODULE_LIBRARY")
    target_link_libraries(${executable} ${cmake_2_8_12_PRIVATE} ${LIBRARIES})
  else()
    # Use plain form for legacy user.
    target_link_libraries(${executable} ${LIBRARIES})
  endif()
endfunction(explicit_llvm_config)


# This is Deprecated
function(llvm_map_components_to_libraries OUT_VAR)
  message(AUTHOR_WARNING "Using llvm_map_components_to_libraries() is deprecated. Use llvm_map_components_to_libnames() instead")
  explicit_map_components_to_libraries(result ${ARGN})
  set( ${OUT_VAR} ${result} ${sys_result} PARENT_SCOPE )
endfunction(llvm_map_components_to_libraries)

# This is a variant intended for the final user:
# Map LINK_COMPONENTS to actual libnames.
function(llvm_map_components_to_libnames out_libs)
  set( link_components ${ARGN} )
  if(NOT LLVM_AVAILABLE_LIBS)
    # Inside LLVM itself available libs are in a global property.
    get_property(LLVM_AVAILABLE_LIBS GLOBAL PROPERTY LLVM_LIBS)
  endif()
  string(TOUPPER "${LLVM_AVAILABLE_LIBS}" capitalized_libs)

  # Expand some keywords:
  list(FIND LLVM_TARGETS_TO_BUILD "${LLVM_NATIVE_ARCH}" have_native_backend)
  list(FIND link_components "engine" engine_required)
  if( NOT engine_required EQUAL -1 )
    list(FIND LLVM_TARGETS_WITH_JIT "${LLVM_NATIVE_ARCH}" have_jit)
    if( NOT have_native_backend EQUAL -1 AND NOT have_jit EQUAL -1 )
      list(APPEND link_components "jit")
      list(APPEND link_components "native")
    else()
      list(APPEND link_components "interpreter")
    endif()
  endif()
  list(FIND link_components "native" native_required)
  if( NOT native_required EQUAL -1 )
    if( NOT have_native_backend EQUAL -1 )
      list(APPEND link_components ${LLVM_NATIVE_ARCH})
    endif()
  endif()

  # Translate symbolic component names to real libraries:
  foreach(c ${link_components})
    # add codegen, asmprinter, asmparser, disassembler
    list(FIND LLVM_TARGETS_TO_BUILD ${c} idx)
    if( NOT idx LESS 0 )
      if( TARGET LLVM${c}CodeGen )
        list(APPEND expanded_components "LLVM${c}CodeGen")
      else()
        if( TARGET LLVM${c} )
          list(APPEND expanded_components "LLVM${c}")
        else()
          message(FATAL_ERROR "Target ${c} is not in the set of libraries.")
        endif()
      endif()
      if( TARGET LLVM${c}AsmPrinter )
        list(APPEND expanded_components "LLVM${c}AsmPrinter")
      endif()
      if( TARGET LLVM${c}AsmParser )
        list(APPEND expanded_components "LLVM${c}AsmParser")
      endif()
      if( TARGET LLVM${c}Desc )
        list(APPEND expanded_components "LLVM${c}Desc")
      endif()
      if( TARGET LLVM${c}Info )
        list(APPEND expanded_components "LLVM${c}Info")
      endif()
      if( TARGET LLVM${c}Disassembler )
        list(APPEND expanded_components "LLVM${c}Disassembler")
      endif()
    elseif( c STREQUAL "native" )
      # already processed
    elseif( c STREQUAL "nativecodegen" )
      list(APPEND expanded_components "LLVM${LLVM_NATIVE_ARCH}CodeGen")
      if( TARGET LLVM${LLVM_NATIVE_ARCH}Desc )
        list(APPEND expanded_components "LLVM${LLVM_NATIVE_ARCH}Desc")
      endif()
      if( TARGET LLVM${LLVM_NATIVE_ARCH}Info )
        list(APPEND expanded_components "LLVM${LLVM_NATIVE_ARCH}Info")
      endif()
    elseif( c STREQUAL "backend" )
      # same case as in `native'.
    elseif( c STREQUAL "engine" )
      # already processed
    elseif( c STREQUAL "all" )
      list(APPEND expanded_components ${LLVM_AVAILABLE_LIBS})
    else( NOT idx LESS 0 )
      # Canonize the component name:
      string(TOUPPER "${c}" capitalized)
      list(FIND capitalized_libs LLVM${capitalized} lib_idx)
      if( lib_idx LESS 0 )
        # The component is unknown. Maybe is an omitted target?
        is_llvm_target_library(${c} iltl_result)
        if( NOT iltl_result )
          message(FATAL_ERROR "Library `${c}' not found in list of llvm libraries.")
        endif()
      else( lib_idx LESS 0 )
        list(GET LLVM_AVAILABLE_LIBS ${lib_idx} canonical_lib)
        list(APPEND expanded_components ${canonical_lib})
      endif( lib_idx LESS 0 )
    endif( NOT idx LESS 0 )
  endforeach(c)

  set(${out_libs} ${expanded_components} PARENT_SCOPE)
endfunction()

# Expand dependencies while topologically sorting the list of libraries:
function(llvm_expand_dependencies out_libs)
  set(expanded_components ${ARGN})
  list(LENGTH expanded_components lst_size)
  set(cursor 0)
  set(processed)
  while( cursor LESS lst_size )
    list(GET expanded_components ${cursor} lib)
    get_property(lib_deps GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_${lib})
    list(APPEND expanded_components ${lib_deps})
    # Remove duplicates at the front:
    list(REVERSE expanded_components)
    list(REMOVE_DUPLICATES expanded_components)
    list(REVERSE expanded_components)
    list(APPEND processed ${lib})
    # Find the maximum index that doesn't have to be re-processed:
    while(NOT "${expanded_components}" MATCHES "^${processed}.*" )
      list(REMOVE_AT processed -1)
    endwhile()
    list(LENGTH processed cursor)
    list(LENGTH expanded_components lst_size)
  endwhile( cursor LESS lst_size )
  set(${out_libs} ${expanded_components} PARENT_SCOPE)
endfunction()

function(explicit_map_components_to_libraries out_libs)
  llvm_map_components_to_libnames(link_libs ${ARGN})
  llvm_expand_dependencies(expanded_components ${link_libs})
  # Return just the libraries included in this build:
  set(result)
  foreach(c ${expanded_components})
    if( TARGET ${c} )
      set(result ${result} ${c})
    endif()
  endforeach(c)
  set(${out_libs} ${result} PARENT_SCOPE)
endfunction(explicit_map_components_to_libraries)
