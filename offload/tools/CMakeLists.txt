set(OFFLOAD_TOOLS_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}" CACHE PATH
    "Path for binary subdirectory (defaults to '${CMAKE_INSTALL_BINDIR}')")
mark_as_advanced(OFFLOAD_TOOLS_INSTALL_DIR)

file(RELATIVE_PATH offload_libdir_relpath
     "${CMAKE_INSTALL_PREFIX}/${OFFLOAD_TOOLS_INSTALL_DIR}"
     "${CMAKE_INSTALL_PREFIX}/${OFFLOAD_INSTALL_LIBDIR}")
set(OFFLOAD_TOOLS_INSTALL_RPATH "$ORIGIN/${offload_libdir_relpath}")
if(LLVM_LINK_LLVM_DYLIB OR LLVM_ENABLE_SHARED_LIBS)
  file(RELATIVE_PATH llvm_libdir_relpath
       "${CMAKE_INSTALL_PREFIX}/${OFFLOAD_TOOLS_INSTALL_DIR}"
       "${CMAKE_INSTALL_PREFIX}/lib${LLVM_LIBDIR_SUFFIX}")
  list(APPEND OFFLOAD_TOOLS_INSTALL_RPATH "$ORIGIN/${llvm_libdir_relpath}")
endif()

set(OFFLOAD_TOOLS_BUILD_RPATH "$ORIGIN/../lib${LLVM_LIBDIR_SUFFIX}")
if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR AND NOT APPLE)
  list(APPEND OFFLOAD_TOOLS_BUILD_RPATH
       "$ORIGIN/../lib${LLVM_LIBDIR_SUFFIX}/${LLVM_DEFAULT_TARGET_TRIPLE}")
endif()

function(add_offload_tool name)
  cmake_parse_arguments(ARG "" "" "LINK_COMPONENTS" ${ARGN})
  add_executable(${name} ${ARG_UNPARSED_ARGUMENTS})
  llvm_update_compile_flags(${name})

  if(ARG_LINK_COMPONENTS)
    if(LLVM_LINK_LLVM_DYLIB)
      set(llvm_libs LLVM)
    else()
      llvm_map_components_to_libnames(llvm_libs ${ARG_LINK_COMPONENTS})
    endif()
    target_link_libraries(${name} PRIVATE ${llvm_libs})
  endif()

  set_target_properties(${name} PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${LLVM_RUNTIME_OUTPUT_INTDIR}"
                        BUILD_RPATH "${OFFLOAD_TOOLS_BUILD_RPATH}"
                        INSTALL_RPATH "${OFFLOAD_TOOLS_INSTALL_RPATH}")

  # Multilib builds share the output and install paths with the base build.
  if(OFFLOAD_LIBDIR_SUBDIR)
    set_target_properties(${name} PROPERTIES EXCLUDE_FROM_ALL ON)
    return()
  endif()

  install(TARGETS ${name}
          COMPONENT offload
          RUNTIME DESTINATION "${OFFLOAD_TOOLS_INSTALL_DIR}")
  add_dependencies(offload ${name})
endfunction()

add_subdirectory(deviceinfo)

# kernelreplay requires libomptarget which is not available on Windows
if(BUILD_LIBOMPTARGET)
  add_subdirectory(kernelreplay)
endif()
