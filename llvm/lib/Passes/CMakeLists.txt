if (MSVC)
  set_source_files_properties(PassBuilder.cpp PROPERTIES COMPILE_FLAGS /bigobj)
endif()

add_llvm_component_library(LLVMPasses
  PassBuilder.cpp
  PassPlugin.cpp
  StandardInstrumentations.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Passes

  DEPENDS
  intrinsics_gen

  LINK_LIBS
  LLVMMLPolicies
  )
