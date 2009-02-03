include(AddFileDependencies)


macro(add_td_sources srcs)
  file(GLOB tds *.td)
  if( tds )
    source_group("TableGen descriptions" FILES ${tds})
    set_source_files_properties(${tds} PROPERTIES HEADER_FILE_ONLY ON)
    list(APPEND ${srcs} ${tds})
  endif()
endmacro(add_td_sources)


macro(add_header_files srcs)
  file(GLOB hds *.h)
  if( hds )
    set_source_files_properties(${hds} PROPERTIES HEADER_FILE_ONLY ON)
    list(APPEND ${srcs} ${hds})
  endif()
endmacro(add_header_files)


function(llvm_process_sources OUT_VAR)
  set( sources ${ARGN} )
  # Create file dependencies on the tablegenned files, if any.  Seems
  # that this is not strictly needed, as dependencies of the .cpp
  # sources on the tablegenned .inc files are detected and handled,
  # but just in case...
  foreach( s ${sources} )
    set( f ${CMAKE_CURRENT_SOURCE_DIR}/${s} )
    add_file_dependencies( ${f} ${TABLEGEN_OUTPUT} )
  endforeach(s)
  if( MSVC_IDE )
    # This adds .td and .h files to the Visual Studio solution:
    add_td_sources(sources)
    add_header_files(sources)
  endif()
  set( ${OUT_VAR} ${sources} PARENT_SCOPE )
endfunction(llvm_process_sources)
