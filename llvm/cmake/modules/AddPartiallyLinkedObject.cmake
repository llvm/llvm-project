include(LLVMProcessSources)

macro(target_name_of_partially_linked_object lib var)
  if( MSVC )
    set(${var} ${lib})
  else( MSVC )
    set(${var} ${lib}_pll)
  endif( MSVC )
endmacro(target_name_of_partially_linked_object lib var)


macro(add_partially_linked_object lib)
  if( MSVC )
    add_llvm_library( ${lib} ${ARGN})
  else( MSVC )
    set(pll ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${lib}.o)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/temp_lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/temp_lib)
    llvm_process_sources( ALL_FILES ${ARGN} )
    if( BUILD_SHARED_LIBS AND SUPPORTS_FPIC_FLAG )
      add_definitions(-fPIC)
    endif()
    add_library( ${lib} STATIC ${ALL_FILES})
    if( LLVM_COMMON_DEPENDS )
      add_dependencies( ${lib} ${LLVM_COMMON_DEPENDS} )
    endif( LLVM_COMMON_DEPENDS )
    add_custom_command(OUTPUT ${pll}
      COMMENT "Building ${lib}.o..."
      DEPENDS ${lib}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/temp_lib
      COMMAND ar x ${CMAKE_STATIC_LIBRARY_PREFIX}${lib}${CMAKE_STATIC_LIBRARY_SUFFIX}
      COMMAND ${CMAKE_LINKER} "${LLVM_PLO_FLAGS}" -r "*${CMAKE_CXX_OUTPUT_EXTENSION}" -o ${pll}
      COMMAND ${CMAKE_COMMAND} -E remove -f *${CMAKE_CXX_OUTPUT_EXTENSION}
      )
    target_name_of_partially_linked_object(${lib} tnplo)
    add_custom_target(${tnplo} ALL DEPENDS ${pll})
    set( llvm_libs ${llvm_libs} ${pll} PARENT_SCOPE)
    set( llvm_lib_targets ${llvm_lib_targets} ${tnplo} PARENT_SCOPE )
  endif( MSVC )
  install(FILES ${pll}
    DESTINATION lib)
endmacro(add_partially_linked_object lib)
