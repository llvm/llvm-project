# Link LibCURL if the user wants it
if (LLVM_ENABLE_CURL)
  set(imported_libs ${CURL_LIBRARIES})
endif()

add_llvm_component_library(LLVMDebuginfod
  Debuginfod.cpp
  HTTPClient.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Debuginfod

  LINK_LIBS
  ${imported_libs}

  LINK_COMPONENTS
  Support
  )

set(llvm_system_libs ${system_libs})
