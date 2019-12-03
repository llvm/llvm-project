set(system_libs)
if( CMAKE_HOST_UNIX )
  if( LLVM_LIBXML2_ENABLED )
    set(system_libs ${system_libs} ${LIBXML2_LIBS})
  endif()
endif()

add_llvm_component_library(LLVMWindowsManifest
  WindowsManifestMerger.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/WindowsManifest
  ${Backtrace_INCLUDE_DIRS}

  LINK_LIBS ${system_libs}
  )

set_property(TARGET LLVMWindowsManifest PROPERTY LLVM_SYSTEM_LIBS "${system_libs}")
