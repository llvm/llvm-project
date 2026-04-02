add_llvm_component_library(LLVMDebugInfoCodeView
  AppendingTypeTableBuilder.cpp
  CodeViewError.cpp
  CodeViewRecordIO.cpp
  ContinuationRecordBuilder.cpp
  CVSymbolVisitor.cpp
  CVTypeVisitor.cpp
  DebugChecksumsSubsection.cpp
  DebugCrossExSubsection.cpp
  DebugCrossImpSubsection.cpp
  DebugFrameDataSubsection.cpp
  DebugInlineeLinesSubsection.cpp
  DebugLinesSubsection.cpp
  DebugStringTableSubsection.cpp
  DebugSubsection.cpp
  DebugSubsectionRecord.cpp
  DebugSubsectionVisitor.cpp
  DebugSymbolRVASubsection.cpp
  DebugSymbolsSubsection.cpp
  EnumTables.cpp
  Formatters.cpp
  GlobalTypeTableBuilder.cpp
  LazyRandomTypeCollection.cpp
  Line.cpp
  MergingTypeTableBuilder.cpp
  RecordName.cpp
  RecordSerialization.cpp
  SimpleTypeSerializer.cpp
  StringsAndChecksums.cpp
  SymbolDumper.cpp
  SymbolRecordHelpers.cpp
  SymbolRecordMapping.cpp
  SymbolSerializer.cpp
  TypeDumpVisitor.cpp
  TypeIndex.cpp
  TypeIndexDiscovery.cpp
  TypeHashing.cpp
  TypeRecordHelpers.cpp
  TypeRecordMapping.cpp
  TypeStreamMerger.cpp
  TypeTableCollection.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/DebugInfo/CodeView

  LINK_COMPONENTS
  Support
  )

# These files define static functions/arrays with identical names (stabilize,
# LeafTypeNames, getLeafTypeName) that conflict in unity builds.
set_source_files_properties(
  AppendingTypeTableBuilder.cpp
  GlobalTypeTableBuilder.cpp
  MergingTypeTableBuilder.cpp
  TypeDumpVisitor.cpp
  TypeRecordMapping.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
