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
#include "llvm/DebugInfo/CodeView/ModuleDebugFileChecksumFragment.h"
#include "llvm/DebugInfo/CodeView/ModuleDebugFragmentVisitor.h"
#include "llvm/Support/Error.h"

#include <vector>

namespace llvm {

namespace pdb {

class PDBFile;

class C13DebugFragmentVisitor : public codeview::ModuleDebugFragmentVisitor {
public:
  C13DebugFragmentVisitor(PDBFile &F);
  ~C13DebugFragmentVisitor();

  Error visitUnknown(codeview::ModuleDebugUnknownFragmentRef &Fragment) final;

  Error visitFileChecksums(
      codeview::ModuleDebugFileChecksumFragmentRef &Checksums) final;

  Error visitLines(codeview::ModuleDebugLineFragmentRef &Lines) final;

  Error
  visitInlineeLines(codeview::ModuleDebugInlineeLineFragmentRef &Lines) final;

  Error finished() final;

protected:
  virtual Error handleFileChecksums() { return Error::success(); }
  virtual Error handleLines() { return Error::success(); }
  virtual Error handleInlineeLines() { return Error::success(); }

  Expected<StringRef> getNameFromStringTable(uint32_t Offset);
  Expected<StringRef> getNameFromChecksumsBuffer(uint32_t Offset);

  Optional<codeview::ModuleDebugFileChecksumFragmentRef> Checksums;
  std::vector<codeview::ModuleDebugInlineeLineFragmentRef> InlineeLines;
  std::vector<codeview::ModuleDebugLineFragmentRef> Lines;

  PDBFile &F;
};
}
}

#endif
