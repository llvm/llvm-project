if(MINGW)
  # CoCreateInstance and the BSTR functions of the Setup API lookup.
  set(imported_libs ole32 oleaut32)
endif()

add_llvm_component_library(LLVMWindowsDriver
  MSVCPaths.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/WindowsDriver

  LINK_LIBS
  ${imported_libs}

  LINK_COMPONENTS
  Option
  Support
  TargetParser
  )
