//===- CodeViewYAMLDebugSections.h - CodeView YAMLIO debug sections -------===//
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

#ifndef LLVM_OBJECTYAML_CODEVIEWYAMLDEBUGSECTIONS_H
#define LLVM_OBJECTYAML_CODEVIEWYAMLDEBUGSECTIONS_H

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/DebugSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugSubsectionRecord.h"
#include "llvm/ObjectYAML/YAML.h"

namespace llvm {

namespace codeview {
class DebugStringTableSubsection;
class DebugStringTableSubsectionRef;
class DebugChecksumsSubsectionRef;
class DebugStringTableSubsection;
class DebugChecksumsSubsection;
class StringsAndChecksums;
class StringsAndChecksumsRef;
}
namespace CodeViewYAML {

namespace detail {
struct YAMLSubsectionBase;
}

struct YAMLFrameData {
  uint32_t RvaStart;
  uint32_t CodeSize;
  uint32_t LocalSize;
  uint32_t ParamsSize;
  uint32_t MaxStackSize;
  StringRef FrameFunc;
  uint32_t PrologSize;
  uint32_t SavedRegsSize;
  uint32_t Flags;
};

struct YAMLCrossModuleImport {
  StringRef ModuleName;
  std::vector<uint32_t> ImportIds;
};

struct SourceLineEntry {
  uint32_t Offset;
  uint32_t LineStart;
  uint32_t EndDelta;
  bool IsStatement;
};

struct SourceColumnEntry {
  uint16_t StartColumn;
  uint16_t EndColumn;
};

struct SourceLineBlock {
  StringRef FileName;
  std::vector<SourceLineEntry> Lines;
  std::vector<SourceColumnEntry> Columns;
};

struct HexFormattedString {
  std::vector<uint8_t> Bytes;
};

struct SourceFileChecksumEntry {
  StringRef FileName;
  codeview::FileChecksumKind Kind;
  HexFormattedString ChecksumBytes;
};

struct SourceLineInfo {
  uint32_t RelocOffset;
  uint32_t RelocSegment;
  codeview::LineFlags Flags;
  uint32_t CodeSize;

  std::vector<SourceLineBlock> Blocks;
};

struct InlineeSite {
  uint32_t Inlinee;
  StringRef FileName;
  uint32_t SourceLineNum;
  std::vector<StringRef> ExtraFiles;
};

struct InlineeInfo {
  bool HasExtraFiles;
  std::vector<InlineeSite> Sites;
};

struct YAMLDebugSubsection {
  static Expected<YAMLDebugSubsection>
  fromCodeViewSubection(const codeview::StringsAndChecksumsRef &SC,
                        const codeview::DebugSubsectionRecord &SS);

  std::shared_ptr<detail::YAMLSubsectionBase> Subsection;
};

struct DebugSubsectionState {};

Expected<std::vector<std::shared_ptr<codeview::DebugSubsection>>>
toCodeViewSubsectionList(BumpPtrAllocator &Allocator,
                         ArrayRef<YAMLDebugSubsection> Subsections,
                         const codeview::StringsAndChecksums &SC);

std::vector<YAMLDebugSubsection>
fromDebugS(ArrayRef<uint8_t> Data, const codeview::StringsAndChecksumsRef &SC);

void initializeStringsAndChecksums(ArrayRef<YAMLDebugSubsection> Sections,
                                   codeview::StringsAndChecksums &SC);

} // namespace CodeViewYAML
} // namespace llvm

LLVM_YAML_DECLARE_MAPPING_TRAITS(CodeViewYAML::YAMLDebugSubsection)

LLVM_YAML_IS_SEQUENCE_VECTOR(CodeViewYAML::YAMLDebugSubsection)

#endif
