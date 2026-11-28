set(LLVM_OPTIONAL_SOURCES
  MSFZ.cpp
  )

if(LLVM_ENABLE_ZSTD)
  set(msfz_source MSFZ.cpp)
endif()

add_llvm_component_library(LLVMDebugInfoMSF
  MappedBlockStream.cpp
  MSFBuilder.cpp
  MSFCommon.cpp
  MSFError.cpp
  ${msfz_source}
  ADDITIONAL_HEADER_DIRS
  "${LLVM_MAIN_INCLUDE_DIR}/llvm/DebugInfo/MSF"

  LINK_COMPONENTS
  Support
  )
