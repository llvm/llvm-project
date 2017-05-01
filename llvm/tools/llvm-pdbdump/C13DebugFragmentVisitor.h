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

  Error visitUnknown(codeview::ModuleDebugUnknownFragment &Fragment) final;

  Error visitFileChecksums(
      codeview::ModuleDebugFileChecksumFragment &Checksums) final;

  Error visitLines(codeview::ModuleDebugLineFragment &Lines) final;

  Error finished() final;

protected:
  virtual Error handleFileChecksums() { return Error::success(); }
  virtual Error handleLines() { return Error::success(); }

  Expected<StringRef> getNameFromStringTable(uint32_t Offset);
  Expected<StringRef> getNameFromChecksumsBuffer(uint32_t Offset);

  Optional<codeview::ModuleDebugFileChecksumFragment> Checksums;
  std::vector<codeview::ModuleDebugLineFragment> Lines;

  PDBFile &F;
};
}
}

#endif
