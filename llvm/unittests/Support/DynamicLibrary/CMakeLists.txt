set(LLVM_LINK_COMPONENTS Support)

add_llvm_unittest(DynamicLibraryTests DynamicLibraryTest.cpp)

export_executable_symbols(DynamicLibraryTests)

add_library(PipSqueak SHARED PipSqueak.cxx)

set_output_directory(PipSqueak
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
  LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
  )

set_target_properties(PipSqueak
  PROPERTIES PREFIX ""
  SUFFIX ".so"
  )

add_dependencies(DynamicLibraryTests PipSqueak)
