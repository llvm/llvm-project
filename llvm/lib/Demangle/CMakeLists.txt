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

# These demangler files each define classes/type aliases named 'Demangler' and
# use "using namespace" directives that bring conflicting symbols (Node,
# Qualifiers, etc.) into scope from different namespaces, which cannot coexist
# in a single translation unit.
set_source_files_properties(
  ItaniumDemangle.cpp
  MicrosoftDemangle.cpp
  RustDemangle.cpp
  DLangDemangle.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
