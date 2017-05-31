//===- CodeViewYAMLDebugSections.cpp - CodeView YAMLIO debug sections -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines classes for handling the YAML representation of CodeView
// Debug Info.
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/CodeViewYAMLDebugSections.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/DebugInfo/CodeView/CodeViewError.h"
#include "llvm/DebugInfo/CodeView/EnumTables.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::CodeViewYAML;
using namespace llvm::CodeViewYAML::detail;
using namespace llvm::yaml;

LLVM_YAML_IS_SEQUENCE_VECTOR(SourceFileChecksumEntry)
LLVM_YAML_IS_SEQUENCE_VECTOR(SourceLineEntry)
LLVM_YAML_IS_SEQUENCE_VECTOR(SourceColumnEntry)
LLVM_YAML_IS_SEQUENCE_VECTOR(SourceLineBlock)
LLVM_YAML_IS_SEQUENCE_VECTOR(SourceLineInfo)
LLVM_YAML_IS_SEQUENCE_VECTOR(InlineeSite)
LLVM_YAML_IS_SEQUENCE_VECTOR(InlineeInfo)
LLVM_YAML_IS_SEQUENCE_VECTOR(StringRef)

LLVM_YAML_DECLARE_SCALAR_TRAITS(HexFormattedString, false)
LLVM_YAML_DECLARE_ENUM_TRAITS(FileChecksumKind)
LLVM_YAML_DECLARE_BITSET_TRAITS(LineFlags)

LLVM_YAML_DECLARE_MAPPING_TRAITS(CodeViewYAML::SourceLineEntry)
LLVM_YAML_DECLARE_MAPPING_TRAITS(CodeViewYAML::SourceColumnEntry)
LLVM_YAML_DECLARE_MAPPING_TRAITS(CodeViewYAML::SourceFileChecksumEntry)
LLVM_YAML_DECLARE_MAPPING_TRAITS(CodeViewYAML::SourceLineInfo)
LLVM_YAML_DECLARE_MAPPING_TRAITS(CodeViewYAML::SourceLineBlock)
LLVM_YAML_DECLARE_MAPPING_TRAITS(CodeViewYAML::InlineeInfo)
LLVM_YAML_DECLARE_MAPPING_TRAITS(CodeViewYAML::InlineeSite)

void ScalarBitSetTraits<LineFlags>::bitset(IO &io, LineFlags &Flags) {
  io.bitSetCase(Flags, "HasColumnInfo", LF_HaveColumns);
  io.enumFallback<Hex16>(Flags);
}

void ScalarEnumerationTraits<FileChecksumKind>::enumeration(
    IO &io, FileChecksumKind &Kind) {
  io.enumCase(Kind, "None", FileChecksumKind::None);
  io.enumCase(Kind, "MD5", FileChecksumKind::MD5);
  io.enumCase(Kind, "SHA1", FileChecksumKind::SHA1);
  io.enumCase(Kind, "SHA256", FileChecksumKind::SHA256);
}

void ScalarTraits<HexFormattedString>::output(const HexFormattedString &Value,
                                              void *ctx, raw_ostream &Out) {
  StringRef Bytes(reinterpret_cast<const char *>(Value.Bytes.data()),
                  Value.Bytes.size());
  Out << toHex(Bytes);
}

StringRef ScalarTraits<HexFormattedString>::input(StringRef Scalar, void *ctxt,
                                                  HexFormattedString &Value) {
  std::string H = fromHex(Scalar);
  Value.Bytes.assign(H.begin(), H.end());
  return StringRef();
}

void MappingTraits<SourceLineEntry>::mapping(IO &IO, SourceLineEntry &Obj) {
  IO.mapRequired("Offset", Obj.Offset);
  IO.mapRequired("LineStart", Obj.LineStart);
  IO.mapRequired("IsStatement", Obj.IsStatement);
  IO.mapRequired("EndDelta", Obj.EndDelta);
}

void MappingTraits<SourceColumnEntry>::mapping(IO &IO, SourceColumnEntry &Obj) {
  IO.mapRequired("StartColumn", Obj.StartColumn);
  IO.mapRequired("EndColumn", Obj.EndColumn);
}

void MappingTraits<SourceLineBlock>::mapping(IO &IO, SourceLineBlock &Obj) {
  IO.mapRequired("FileName", Obj.FileName);
  IO.mapRequired("Lines", Obj.Lines);
  IO.mapRequired("Columns", Obj.Columns);
}

void MappingTraits<SourceFileChecksumEntry>::mapping(
    IO &IO, SourceFileChecksumEntry &Obj) {
  IO.mapRequired("FileName", Obj.FileName);
  IO.mapRequired("Kind", Obj.Kind);
  IO.mapRequired("Checksum", Obj.ChecksumBytes);
}

void MappingTraits<SourceLineInfo>::mapping(IO &IO, SourceLineInfo &Obj) {
  IO.mapRequired("CodeSize", Obj.CodeSize);

  IO.mapRequired("Flags", Obj.Flags);
  IO.mapRequired("RelocOffset", Obj.RelocOffset);
  IO.mapRequired("RelocSegment", Obj.RelocSegment);
  IO.mapRequired("Blocks", Obj.Blocks);
}

void MappingTraits<SourceFileInfo>::mapping(IO &IO, SourceFileInfo &Obj) {
  IO.mapOptional("Checksums", Obj.FileChecksums);
  IO.mapOptional("Lines", Obj.LineFragments);
  IO.mapOptional("InlineeLines", Obj.Inlinees);
}

void MappingTraits<InlineeSite>::mapping(IO &IO, InlineeSite &Obj) {
  IO.mapRequired("FileName", Obj.FileName);
  IO.mapRequired("LineNum", Obj.SourceLineNum);
  IO.mapRequired("Inlinee", Obj.Inlinee);
  IO.mapOptional("ExtraFiles", Obj.ExtraFiles);
}

void MappingTraits<InlineeInfo>::mapping(IO &IO, InlineeInfo &Obj) {
  IO.mapRequired("HasExtraFiles", Obj.HasExtraFiles);
  IO.mapRequired("Sites", Obj.Sites);
}
