function(get_system_libs return_var)
  # Returns in `return_var' a list of system libraries used by LLVM.
  if( NOT MSVC )
    if( MINGW )
      set(system_libs ${system_libs} imagehlp psapi)
    elseif( CMAKE_HOST_UNIX )
      if( HAVE_LIBDL )
	set(system_libs ${system_libs} ${CMAKE_DL_LIBS})
      endif()
      if( LLVM_ENABLE_THREADS AND HAVE_LIBPTHREAD )
	set(system_libs ${system_libs} pthread)
      endif()
    endif( MINGW )
  endif( NOT MSVC )
  set(${return_var} ${system_libs} PARENT_SCOPE)
endfunction(get_system_libs)


function(link_system_libs target)
  get_system_libs(llvm_system_libs)
  target_link_libraries(${target} ${llvm_system_libs})
endfunction(link_system_libs)


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

  explicit_map_components_to_libraries(LIBRARIES ${link_components})
  target_link_libraries(${executable} ${LIBRARIES})
endfunction(explicit_llvm_config)


# This is a variant intended for the final user:
function(llvm_map_components_to_libraries OUT_VAR)
  explicit_map_components_to_libraries(result ${ARGN})
  get_system_libs(sys_result)
  set( ${OUT_VAR} ${result} ${sys_result} PARENT_SCOPE )
endfunction(llvm_map_components_to_libraries)


function(explicit_map_components_to_libraries out_libs)
  set( link_components ${ARGN} )
  get_property(llvm_libs GLOBAL PROPERTY LLVM_LIBS)
  string(TOUPPER "${llvm_libs}" capitalized_libs)

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
      list(FIND llvm_libs "LLVM${c}CodeGen" idx)
      if( NOT idx LESS 0 )
	list(APPEND expanded_components "LLVM${c}CodeGen")
      else()
	list(FIND llvm_libs "LLVM${c}" idx)
	if( NOT idx LESS 0 )
	  list(APPEND expanded_components "LLVM${c}")
	else()
	  message(FATAL_ERROR "Target ${c} is not in the set of libraries.")
	endif()
      endif()
      list(FIND llvm_libs "LLVM${c}AsmPrinter" asmidx)
      if( NOT asmidx LESS 0 )
        list(APPEND expanded_components "LLVM${c}AsmPrinter")
      endif()
      list(FIND llvm_libs "LLVM${c}AsmParser" asmidx)
      if( NOT asmidx LESS 0 )
        list(APPEND expanded_components "LLVM${c}AsmParser")
      endif()
      list(FIND llvm_libs "LLVM${c}Info" asmidx)
      if( NOT asmidx LESS 0 )
        list(APPEND expanded_components "LLVM${c}Info")
      endif()
      list(FIND llvm_libs "LLVM${c}Disassembler" asmidx)
      if( NOT asmidx LESS 0 )
        list(APPEND expanded_components "LLVM${c}Disassembler")
      endif()
    elseif( c STREQUAL "native" )
      # already processed
    elseif( c STREQUAL "nativecodegen" )
      list(APPEND expanded_components "LLVM${LLVM_NATIVE_ARCH}CodeGen")
    elseif( c STREQUAL "backend" )
      # same case as in `native'.
    elseif( c STREQUAL "engine" )
      # already processed
    elseif( c STREQUAL "all" )
      list(APPEND expanded_components ${llvm_libs})
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
        list(GET llvm_libs ${lib_idx} canonical_lib)
        list(APPEND expanded_components ${canonical_lib})
      endif( lib_idx LESS 0 )
    endif( NOT idx LESS 0 )
  endforeach(c)
  # Expand dependencies while topologically sorting the list of libraries:
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
  # Return just the libraries included in this build:
  set(result)
  foreach(c ${expanded_components})
    list(FIND llvm_libs ${c} lib_idx)
    if( NOT lib_idx LESS 0 )
      set(result ${result} ${c})
    endif()
  endforeach(c)
  set(${out_libs} ${result} PARENT_SCOPE)
endfunction(explicit_map_components_to_libraries)
