set(LLVM_NO_RTTI 1)

add_lldb_library(lldbPluginSymbolFileDWARF
  DWARFAbbreviationDeclaration.cpp
  DWARFCompileUnit.cpp
  DWARFDebugAbbrev.cpp
  DWARFDebugAranges.cpp
  DWARFDebugArangeSet.cpp
  DWARFDebugInfo.cpp
  DWARFDebugInfoEntry.cpp
  DWARFDebugLine.cpp
  DWARFDebugMacinfo.cpp
  DWARFDebugMacinfoEntry.cpp
  DWARFDebugPubnames.cpp
  DWARFDebugPubnamesSet.cpp
  DWARFDebugRanges.cpp
  DWARFDeclContext.cpp
  DWARFDefines.cpp
  DWARFDIECollection.cpp
  DWARFFormValue.cpp
  DWARFLocationDescription.cpp
  DWARFLocationList.cpp
  LogChannelDWARF.cpp
  NameToDIE.cpp
  SymbolFileDWARF.cpp
  SymbolFileDWARFDebugMap.cpp
  UniqueDWARFASTType.cpp
  )
