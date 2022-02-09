# Link LibCURL if the user wants it
if (LLVM_ENABLE_CURL)
  set(imported_libs CURL::libcurl)
endif()

# Note: This isn't a component, since that could potentially add a libcurl
# dependency to libLLVM.
add_llvm_library(LLVMDebuginfod
  Debuginfod.cpp
  DIFetcher.cpp
  HTTPClient.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Debuginfod

  LINK_LIBS
  ${imported_libs}

  LINK_COMPONENTS
  Support
  )
