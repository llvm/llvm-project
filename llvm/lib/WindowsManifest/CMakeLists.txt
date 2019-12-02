add_llvm_component_library(LLVMWindowsManifest
  WindowsManifestMerger.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/WindowsManifest
  ${Backtrace_INCLUDE_DIRS}

  LINK_LIBS ${LIBXML2_LIBRARIES}
  )

set_property(TARGET LLVMWindowsManifest PROPERTY
  LLVM_SYSTEM_LIBS ${LIBXML2_LIBRARIES})
