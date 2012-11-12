# Discover the projects that use CMake in the subdirectories.
# Note that explicit cmake invocation is required every time a new project is
# added or removed.
file(GLOB entries *)
foreach(entry ${entries})
  if(IS_DIRECTORY ${entry} AND EXISTS ${entry}/CMakeLists.txt)
    if((NOT ${entry} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR}/compiler-rt) AND
       (NOT ${entry} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR}/libcxx))
      add_subdirectory(${entry})
    endif()
  endif()
endforeach(entry)

# Also add in the compiler-rt tree if present and we have a sufficiently
# recent version of CMake.
if(${CMAKE_VERSION} VERSION_GREATER 2.8.7 AND
   ${LLVM_BUILD_RUNTIME})
  add_llvm_external_project(compiler-rt)
endif()
