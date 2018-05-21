# Needed by LLVM's CMake checks because this file defines multiple targets.
set(LLVM_OPTIONAL_SOURCES ExportedFuncs.cpp PipSqueak.cpp)

set(LLVM_LINK_COMPONENTS Support)

add_library(DynamicLibraryLib STATIC
  ExportedFuncs.cpp
  )
set_target_properties(DynamicLibraryLib PROPERTIES FOLDER "Tests")

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
    SUFFIX ${LTDL_SHLIB_EXT}
    )

  add_dependencies(DynamicLibraryTests ${NAME})
endfunction(dynlib_add_module)

dynlib_add_module(PipSqueak)
dynlib_add_module(SecondLib)
