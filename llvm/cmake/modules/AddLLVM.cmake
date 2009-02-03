include(LLVMProcessSources)
include(LLVMConfig)

macro(add_llvm_library name)
  llvm_process_sources( ALL_FILES ${ARGN} )
  add_library( ${name} ${ALL_FILES} )
  set( llvm_libs ${llvm_libs} ${name} PARENT_SCOPE)
  set( llvm_lib_targets ${llvm_lib_targets} ${name} PARENT_SCOPE )
  if( LLVM_COMMON_DEPENDS )
    add_dependencies( ${name} ${LLVM_COMMON_DEPENDS} )
  endif( LLVM_COMMON_DEPENDS )
  install(TARGETS ${name}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
endmacro(add_llvm_library name)


macro(add_llvm_executable name)
  llvm_process_sources( ALL_FILES ${ARGN} )
  add_executable(${name} ${ALL_FILES})
  if( LLVM_USED_LIBS )
    foreach(lib ${LLVM_USED_LIBS})
      target_link_libraries( ${name} ${lib} )
    endforeach(lib)
  endif( LLVM_USED_LIBS )
  if( LLVM_LINK_COMPONENTS )
    llvm_config(${name} ${LLVM_LINK_COMPONENTS})
  endif( LLVM_LINK_COMPONENTS )
  if( MSVC )
    target_link_libraries(${name} ${llvm_libs})
  else( MSVC )
    add_dependencies(${name} llvm-config.target)
    if( MINGW )
      target_link_libraries(${name} imagehlp psapi)
    elseif( CMAKE_HOST_UNIX )
      target_link_libraries(${name} dl)
    endif( MINGW )
  endif( MSVC )
endmacro(add_llvm_executable name)


macro(add_llvm_tool name)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LLVM_TOOLS_BINARY_DIR})
  add_llvm_executable(${name} ${ARGN})
  install(TARGETS ${name}
    RUNTIME DESTINATION bin)
endmacro(add_llvm_tool name)


macro(add_llvm_example name)
#  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LLVM_EXAMPLES_BINARY_DIR})
  add_llvm_executable(${name} ${ARGN})
  install(TARGETS ${name}
    RUNTIME DESTINATION examples)
endmacro(add_llvm_example name)


macro(add_llvm_target target_name)
  if( TABLEGEN_OUTPUT )
    add_custom_target(${target_name}Table_gen
      DEPENDS ${TABLEGEN_OUTPUT})
    add_dependencies(${target_name}Table_gen ${LLVM_COMMON_DEPENDS})
  endif( TABLEGEN_OUTPUT )
  include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR})
  add_partially_linked_object(LLVM${target_name} ${ARGN})
  if( TABLEGEN_OUTPUT )
    add_dependencies(LLVM${target_name} ${target_name}Table_gen)
  endif( TABLEGEN_OUTPUT )
endmacro(add_llvm_target)
