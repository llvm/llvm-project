include(GetLibraryName)

if(LLVM_ENABLE_LIBXML2 AND NOT LLVM_USE_STATIC_LIBXML2)
  set(imported_libs LibXml2::LibXml2)
endif()

add_llvm_component_library(LLVMWindowsManifest
  WindowsManifestMerger.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/WindowsManifest
  ${Backtrace_INCLUDE_DIRS}

  LINK_LIBS
  ${imported_libs}

  LINK_COMPONENTS
  Support
  )

if(LLVM_ENABLE_LIBXML2 AND LLVM_USE_STATIC_LIBXML2)
  target_link_libraries(LLVMWindowsManifest PRIVATE ${LIBXML2_LIBRARIES})
  target_include_directories(LLVMWindowsManifest PRIVATE ${LIBXML2_INCLUDE_DIRS})
  add_dependencies(LLVMWindowsManifest libxml2)
endif()

# This block is only needed for llvm-config. When we deprecate llvm-config and
# move to using CMake export, this block can be removed.
if(LLVM_ENABLE_LIBXML2)
  if(LLVM_USE_STATIC_LIBXML2)
    # When using static libxml2 built via ExternalProject, use the library path directly
    get_library_name(${LIBXML2_LIBRARIES} libxml2_library)
  else()
    # CMAKE_BUILD_TYPE is only meaningful to single-configuration generators.
    if(CMAKE_BUILD_TYPE)
      string(TOUPPER ${CMAKE_BUILD_TYPE} build_type)
      get_property(libxml2_library TARGET LibXml2::LibXml2 PROPERTY LOCATION_${build_type})
    endif()
    if(NOT libxml2_library)
      get_property(libxml2_library TARGET LibXml2::LibXml2 PROPERTY LOCATION)
    endif()
    get_library_name(${libxml2_library} libxml2_library)
  endif()
  set_property(TARGET LLVMWindowsManifest PROPERTY LLVM_SYSTEM_LIBS ${libxml2_library})
endif()
