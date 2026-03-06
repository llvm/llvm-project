# Make sure pthread is linked if this is a unix host
if (CMAKE_HOST_UNIX)
  set(imported_libs ${imported_libs} ${LLVM_PTHREAD_LIB})
endif()

# Note: This isn't a component, since that could potentially add a libcurl
# dependency to libLLVM.
add_llvm_library(LLVMDebuginfod
  BuildIDFetcher.cpp
  Debuginfod.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Debuginfod

  LINK_LIBS
  ${imported_libs}

  LINK_COMPONENTS
  Support
  SupportHTTP
  Symbolize
  DebugInfoDWARF
  BinaryFormat
  Object
  )
