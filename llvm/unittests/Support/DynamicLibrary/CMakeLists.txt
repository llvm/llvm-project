# Needed by LLVM's CMake checks because this file defines multiple targets.
set(LLVM_OPTIONAL_SOURCES ExportedFuncs.cpp PipSqueak.cpp)

set(LLVM_LINK_COMPONENTS Support)

add_library(DynamicLibraryLib STATIC
  ExportedFuncs.cpp
  )
set_target_properties(DynamicLibraryLib PROPERTIES FOLDER "Tests")

# extract_symbols.py relies on all its library arguments being in the same
# directory, so we must set the output directory in the same way as if
# add_llvm_library was used.
set_output_directory(DynamicLibraryLib
  LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR}
  )

add_llvm_unittest(DynamicLibraryTests
  DynamicLibraryTest.cpp
  )
target_link_libraries(DynamicLibraryTests PRIVATE DynamicLibraryLib)
export_executable_symbols(DynamicLibraryTests)

function(dynlib_add_module NAME)
  add_library(${NAME} MODULE
    PipSqueak.cpp
    )
  set_target_properties(${NAME} PROPERTIES FOLDER "Tests")

  set_output_directory(${NAME}
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
    LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
    )

  set_target_properties(${NAME}
    PROPERTIES PREFIX ""
    SUFFIX ${LLVM_PLUGIN_EXT}
    )

  add_dependencies(DynamicLibraryTests ${NAME})
  
  if(LLVM_INTEGRATED_CRT_ALLOC)
    # We need to link in the Support lib for the Memory allocator override,
    # otherwise the DynamicLibrary.Shutdown test will fail, because it would
    # allocate memory with the CRT allocator, and release it with our custom
    # allocator (see llvm/lib/Support/Windows/Memory.inc).
    # /INCLUDE:malloc is there to force searching into LLVMSupport before libucrt
    llvm_map_components_to_libnames(llvm_libs Support)
    target_link_libraries(${NAME} ${llvm_libs} "-INCLUDE:malloc")
  endif()
  
endfunction(dynlib_add_module)

# Revert -Wl,-z,nodelete on this test since it relies on the file
# being unloaded.
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  string(REPLACE "-Wl,-z,nodelete" "" CMAKE_MODULE_LINKER_FLAGS
    ${CMAKE_MODULE_LINKER_FLAGS})
endif()

dynlib_add_module(PipSqueak)
dynlib_add_module(SecondLib)
