# Link LibCURL if the user wants it
if (LLVM_ENABLE_CURL)
  set(imported_libs CURL::libcurl)
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

# This block is only needed for llvm-config. When we deprecate llvm-config and
# move to using CMake export, this block can be removed.
if(LLVM_ENABLE_CURL)
  # CMAKE_BUILD_TYPE is only meaningful to single-configuration generators.
  if(CMAKE_BUILD_TYPE)
    string(TOUPPER ${CMAKE_BUILD_TYPE} build_type)
    get_property(curl_library TARGET CURL::libcurl PROPERTY LOCATION_${build_type})
  endif()
  if(NOT curl_library)
    get_property(curl_library TARGET CURL::libcurl PROPERTY LOCATION)
  endif()
  get_library_name(${curl_library} curl_library)
  set(llvm_system_libs ${llvm_system_libs} "${curl_library}")
endif()
