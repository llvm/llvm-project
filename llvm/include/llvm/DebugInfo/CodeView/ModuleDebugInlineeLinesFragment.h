//===- ModuleDebugInlineeLinesFragment.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_MODULEDEBUGINLINEELINESFRAGMENT_H
#define LLVM_DEBUGINFO_CODEVIEW_MODULEDEBUGINLINEELINESFRAGMENT_H

#include "llvm/DebugInfo/CodeView/Line.h"
#include "llvm/DebugInfo/CodeView/ModuleDebugFragment.h"
#include "llvm/Support/BinaryStreamArray.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace codeview {

class ModuleDebugInlineeLineFragmentRef;

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
  typedef codeview::ModuleDebugInlineeLineFragmentRef ContextType;

  static Error extract(BinaryStreamRef Stream, uint32_t &Len,
                       codeview::InlineeSourceLine &Item,
                       ContextType *Fragment);
};

namespace codeview {
class ModuleDebugInlineeLineFragmentRef final : public ModuleDebugFragmentRef {
  typedef VarStreamArray<InlineeSourceLine> LinesArray;
  typedef LinesArray::Iterator Iterator;

public:
  ModuleDebugInlineeLineFragmentRef();

  static bool classof(const ModuleDebugFragmentRef *S) {
    return S->kind() == ModuleDebugFragmentKind::InlineeLines;
  }

  Error initialize(BinaryStreamReader Reader);
  bool hasExtraFiles() const;

  Iterator begin() const { return Lines.begin(); }
  Iterator end() const { return Lines.end(); }

private:
  InlineeLinesSignature Signature;
  VarStreamArray<InlineeSourceLine> Lines;
};

class ModuleDebugInlineeLineFragment final : public ModuleDebugFragment {
public:
  explicit ModuleDebugInlineeLineFragment(bool HasExtraFiles);

  static bool classof(const ModuleDebugFragment *S) {
    return S->kind() == ModuleDebugFragmentKind::InlineeLines;
  }

  Error commit(BinaryStreamWriter &Writer) override;
  uint32_t calculateSerializedLength() override;

  void addInlineSite(TypeIndex FuncId, uint32_t FileOffset,
                     uint32_t SourceLine);
  void addExtraFile(uint32_t FileOffset);

private:
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
