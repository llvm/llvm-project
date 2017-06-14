set(LLVM_LINK_COMPONENTS Support)

add_library(DynamicLibraryLib STATIC ExportedFuncs.cxx)

add_llvm_unittest(DynamicLibraryTests DynamicLibraryTest.cpp)
target_link_libraries(DynamicLibraryTests DynamicLibraryLib)
export_executable_symbols(DynamicLibraryTests)

function(dynlib_add_module NAME)
  add_library(${NAME} SHARED PipSqueak.cxx)

  set_output_directory(${NAME}
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
    LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
    )

  set_target_properties(${NAME}
    PROPERTIES PREFIX ""
    SUFFIX ".so"
    )

  add_dependencies(DynamicLibraryTests ${NAME})
endfunction(dynlib_add_module)

dynlib_add_module(PipSqueak)
dynlib_add_module(SecondLib)
