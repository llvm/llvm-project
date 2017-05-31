//===- DebugSubsectionVisitor.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_MODULEDEBUGFRAGMENTVISITOR_H
#define LLVM_DEBUGINFO_CODEVIEW_MODULEDEBUGFRAGMENTVISITOR_H

#include "llvm/Support/Error.h"
#include <cstdint>

namespace llvm {

namespace codeview {

class DebugChecksumsSubsectionRef;
class DebugSubsectionRecord;
class DebugInlineeLinesSubsectionRef;
class DebugLinesSubsectionRef;
class DebugUnknownSubsectionRef;

class DebugSubsectionVisitor {
public:
  virtual ~DebugSubsectionVisitor() = default;

  virtual Error visitUnknown(DebugUnknownSubsectionRef &Unknown) {
    return Error::success();
  }
  virtual Error visitLines(DebugLinesSubsectionRef &Lines) {
    return Error::success();
  }

  virtual Error visitFileChecksums(DebugChecksumsSubsectionRef &Checksums) {
    return Error::success();
  }

  virtual Error visitInlineeLines(DebugInlineeLinesSubsectionRef &Inlinees) {
    return Error::success();
  }

  virtual Error finished() { return Error::success(); }
};

Error visitDebugSubsection(const DebugSubsectionRecord &R,
                           DebugSubsectionVisitor &V);

template <typename T>
Error visitDebugSubsections(T &&FragmentRange, DebugSubsectionVisitor &V) {
  for (const auto &L : FragmentRange) {
    if (auto EC = visitDebugSubsection(L, V))
      return EC;
  }
  if (auto EC = V.finished())
    return EC;
  return Error::success();
}

} // end namespace codeview

} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_MODULEDEBUGFRAGMENTVISITOR_H
