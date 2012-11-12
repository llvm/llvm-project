include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. 
                     ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMMBlazeDisassembler
  MBlazeDisassembler.cpp
  )

# workaround for hanging compilation on MSVC9 and 10
if( MSVC_VERSION EQUAL 1500 OR MSVC_VERSION EQUAL 1600 )
set_property(
  SOURCE MBlazeDisassembler.cpp
  PROPERTY COMPILE_FLAGS "/Od"
  )
endif()

add_dependencies(LLVMMBlazeDisassembler MBlazeCommonTableGen)
