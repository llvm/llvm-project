add_llvm_component_library(LLVMTextAPI
  Architecture.cpp
  ArchitectureSet.cpp
  InterfaceFile.cpp
  TextStubV5.cpp
  PackedVersion.cpp
  Platform.cpp
  RecordsSlice.cpp
  RecordVisitor.cpp
  Symbol.cpp
  SymbolSet.cpp
  Target.cpp
  TextAPIError.cpp
  TextStub.cpp
  TextStubCommon.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  "${LLVM_MAIN_INCLUDE_DIR}/llvm/TextAPI"

  LINK_COMPONENTS
  Support
  BinaryFormat
  TargetParser
)

# RecordsSlice.cpp has ambiguous 'Target' due to namespace conflicts in
# unity builds.
set_source_files_properties(RecordsSlice.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

add_subdirectory(BinaryReader)
