add_llvm_component_library(LLVMDemangle
  Demangle.cpp
  ItaniumDemangle.cpp
  MicrosoftDemangle.cpp
  MicrosoftDemangleNodes.cpp
  RustDemangle.cpp
  DLangDemangle.cpp

  ADDITIONAL_HEADER_DIRS
  "${LLVM_MAIN_INCLUDE_DIR}/llvm/Demangle"

)

set_target_properties(LLVMDemangle PROPERTIES UNITY_BUILD_BATCH_SIZE 1)