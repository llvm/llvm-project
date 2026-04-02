set(LLVM_LINK_COMPONENTS
  BinaryFormat
  DebugInfoCodeView
  DebugInfoMSF
  DebugInfoPDB
  Object
  ObjectYAML
  Support
  )

# DumpOutputStyle.cpp defines printHeader which conflicts with identically-named
# functions from other output style files in unity builds.
set_source_files_properties(
  DumpOutputStyle.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON
)

add_llvm_tool(llvm-pdbutil
  BytesOutputStyle.cpp
  DumpOutputStyle.cpp
  ExplainOutputStyle.cpp
  llvm-pdbutil.cpp
  MinimalSymbolDumper.cpp
  MinimalTypeDumper.cpp
  PdbYaml.cpp
  PrettyBuiltinDumper.cpp
  PrettyClassDefinitionDumper.cpp
  PrettyClassLayoutGraphicalDumper.cpp
  PrettyCompilandDumper.cpp
  PrettyEnumDumper.cpp
  PrettyExternalSymbolDumper.cpp
  PrettyFunctionDumper.cpp
  PrettyTypeDumper.cpp
  PrettyTypedefDumper.cpp
  PrettyVariableDumper.cpp
  StreamUtil.cpp
  TypeReferenceTracker.cpp
  YAMLOutputStyle.cpp
  )
