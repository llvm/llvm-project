//===- C13DebugFragmentVisitor.h - Visitor for CodeView Info ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_C13DEBUGFRAGMENTVISITOR_H
#define LLVM_TOOLS_LLVMPDBDUMP_C13DEBUGFRAGMENTVISITOR_H

#include "llvm/ADT/Optional.h"
#include "llvm/DebugInfo/CodeView/DebugChecksumsSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugSubsectionVisitor.h"
#include "llvm/Support/Error.h"

#include <vector>

namespace llvm {

namespace pdb {

class PDBFile;

class C13DebugFragmentVisitor : public codeview::DebugSubsectionVisitor {
public:
  C13DebugFragmentVisitor(PDBFile &F);
  ~C13DebugFragmentVisitor();

  Error visitUnknown(codeview::DebugUnknownSubsectionRef &Fragment) final;

  Error
  visitFileChecksums(codeview::DebugChecksumsSubsectionRef &Checksums) final;

  Error visitLines(codeview::DebugLinesSubsectionRef &Lines) final;

  Error
  visitInlineeLines(codeview::DebugInlineeLinesSubsectionRef &Lines) final;

  Error finished() final;

protected:
  virtual Error handleFileChecksums() { return Error::success(); }
  virtual Error handleLines() { return Error::success(); }
  virtual Error handleInlineeLines() { return Error::success(); }

  Expected<StringRef> getNameFromStringTable(uint32_t Offset);
  Expected<StringRef> getNameFromChecksumsBuffer(uint32_t Offset);

  Optional<codeview::DebugChecksumsSubsectionRef> Checksums;
  std::vector<codeview::DebugInlineeLinesSubsectionRef> InlineeLines;
  std::vector<codeview::DebugLinesSubsectionRef> Lines;

  PDBFile &F;
};
}
}

#endif
