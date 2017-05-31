//===- DebugInlineeLinesSubsection.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_BUGINLINEELINESSUBSECTION_H
#define LLVM_DEBUGINFO_CODEVIEW_BUGINLINEELINESSUBSECTION_H

#include "llvm/DebugInfo/CodeView/DebugSubsection.h"
#include "llvm/DebugInfo/CodeView/Line.h"
#include "llvm/Support/BinaryStreamArray.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace codeview {

class DebugInlineeLinesSubsectionsRef;
class DebugChecksumsSubsection;

enum class InlineeLinesSignature : uint32_t {
  Normal,    // CV_INLINEE_SOURCE_LINE_SIGNATURE
  ExtraFiles // CV_INLINEE_SOURCE_LINE_SIGNATURE_EX
};

struct InlineeSourceLineHeader {
  TypeIndex Inlinee;                  // ID of the function that was inlined.
  support::ulittle32_t FileID;        // Offset into FileChecksums subsection.
  support::ulittle32_t SourceLineNum; // First line of inlined code.
                                      // If extra files present:
                                      //   ulittle32_t ExtraFileCount;
                                      //   ulittle32_t Files[];
};

struct InlineeSourceLine {
  const InlineeSourceLineHeader *Header;
  FixedStreamArray<support::ulittle32_t> ExtraFiles;
};
}

template <> struct VarStreamArrayExtractor<codeview::InlineeSourceLine> {
  typedef bool ContextType;

  static Error extract(BinaryStreamRef Stream, uint32_t &Len,
                       codeview::InlineeSourceLine &Item, bool HasExtraFiles);
};

namespace codeview {
class DebugInlineeLinesSubsectionRef final : public DebugSubsectionRef {
  typedef VarStreamArray<InlineeSourceLine> LinesArray;
  typedef LinesArray::Iterator Iterator;

public:
  DebugInlineeLinesSubsectionRef();

  static bool classof(const DebugSubsectionRef *S) {
    return S->kind() == DebugSubsectionKind::InlineeLines;
  }

  Error initialize(BinaryStreamReader Reader);
  bool hasExtraFiles() const;

  Iterator begin() const { return Lines.begin(); }
  Iterator end() const { return Lines.end(); }

private:
  InlineeLinesSignature Signature;
  VarStreamArray<InlineeSourceLine> Lines;
};

class DebugInlineeLinesSubsection final : public DebugSubsection {
public:
  DebugInlineeLinesSubsection(DebugChecksumsSubsection &Checksums,
                              bool HasExtraFiles);

  static bool classof(const DebugSubsection *S) {
    return S->kind() == DebugSubsectionKind::InlineeLines;
  }

  Error commit(BinaryStreamWriter &Writer) const override;
  uint32_t calculateSerializedSize() const override;

  void addInlineSite(TypeIndex FuncId, StringRef FileName, uint32_t SourceLine);
  void addExtraFile(StringRef FileName);

private:
  DebugChecksumsSubsection &Checksums;

  bool HasExtraFiles = false;
  uint32_t ExtraFileCount = 0;

  struct Entry {
    std::vector<support::ulittle32_t> ExtraFiles;
    InlineeSourceLineHeader Header;
  };
  std::vector<Entry> Entries;
};
}
}

#endif
