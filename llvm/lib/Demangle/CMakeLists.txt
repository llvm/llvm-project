set(LLVMDEMANGLE_SOURCES
  Demangle.cpp
  ItaniumDemangle.cpp
  )

set(LLVMDEMANGLE_OPT_SOURCES
  MicrosoftDemangle.cpp
  MicrosoftDemangleNodes.cpp
  RustDemangle.cpp
  DLangDemangle.cpp
  )

if(NOT EJIT_BARE_METAL)
  list(APPEND LLVMDEMANGLE_SOURCES ${LLVMDEMANGLE_OPT_SOURCES})
endif()

add_llvm_component_library(LLVMDemangle
  ${LLVMDEMANGLE_SOURCES}

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  "${LLVM_MAIN_INCLUDE_DIR}/llvm/Demangle"

)
