add_llvm_component_library(LLVMObjectYAML
  ArchiveEmitter.cpp
  ArchiveYAML.cpp
  CodeViewYAMLDebugSections.cpp
  CodeViewYAMLSymbols.cpp
  CodeViewYAMLTypeHashing.cpp
  CodeViewYAMLTypes.cpp
  COFFEmitter.cpp
  COFFYAML.cpp
  DWARFEmitter.cpp
  DWARFYAML.cpp
  DXContainerEmitter.cpp
  DXContainerYAML.cpp
  ELFEmitter.cpp
  ELFYAML.cpp
  GOFFEmitter.cpp
  GOFFYAML.cpp
  MachOEmitter.cpp
  MachOYAML.cpp
  ObjectYAML.cpp
  MinidumpEmitter.cpp
  MinidumpYAML.cpp
  OffloadEmitter.cpp
  OffloadYAML.cpp
  WasmEmitter.cpp
  WasmYAML.cpp
  XCOFFEmitter.cpp
  XCOFFYAML.cpp
  YAML.cpp
  yaml2obj.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ObjectYAML

  LINK_COMPONENTS
  BinaryFormat
  Object
  Support
  TargetParser
  DebugInfoCodeView
  MC
  )

# Conflicting 'using namespace' declarations (llvm::codeview,
# llvm::CodeViewYAML) cause ambiguity.
set_source_files_properties(
  CodeViewYAMLTypes.cpp
  MinidumpEmitter.cpp
  MinidumpYAML.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
