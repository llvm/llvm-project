add_llvm_component_library(LLVMWindowsManifest
  WindowsManifestMerger.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/WindowsManifest
  ${Backtrace_INCLUDE_DIRS})

if(LIBXML2_LIBRARIES)
  target_link_libraries(LLVMWindowsManifest PUBLIC ${LIBXML2_LIBRARIES})

  get_filename_component(xml2_library ${LIBXML2_LIBRARIES} NAME)
  if(xml2_library MATCHES "^${CMAKE_STATIC_LIBRARY_PREFIX}.*${CMAKE_STATIC_LIBRARY_SUFFIX}$")
    string(REGEX REPLACE "^${CMAKE_STATIC_LIBRARY_PREFIX}" "" xml2_library ${xml2_library})
    string(REGEX REPLACE "${CMAKE_STATIC_LIBRARY_SUFFIX}$" "" xml2_library ${xml2_library})
  elseif(xml2_library MATCHES "^${CMAKE_SHARED_LIBRARY_PREFIX}.*${CMAKE_SHARED_LIBRARY_SUFFIX}$")
    string(REGEX REPLACE "^${CMAKE_SHARED_LIBRARY_PREFIX}" "" xml2_library ${xml2_library})
    string(REGEX REPLACE "${CMAKE_SHARED_LIBRARY_SUFFIX}$" "" xml2_library ${xml2_library})
  endif()
  set_property(TARGET LLVMWindowsManifest PROPERTY
    LLVM_SYSTEM_LIBS ${xml2_library})
endif()
