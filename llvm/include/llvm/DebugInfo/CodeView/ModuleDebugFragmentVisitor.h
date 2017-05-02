//===- ModuleDebugFragmentVisitor.h -----------------------------*- C++ -*-===//
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

class ModuleDebugFileChecksumFragmentRef;
class ModuleDebugFragmentRecord;
class ModuleDebugInlineeLineFragmentRef;
class ModuleDebugLineFragmentRef;
class ModuleDebugUnknownFragmentRef;

class ModuleDebugFragmentVisitor {
public:
  virtual ~ModuleDebugFragmentVisitor() = default;

  virtual Error visitUnknown(ModuleDebugUnknownFragmentRef &Unknown) {
    return Error::success();
  }
  virtual Error visitLines(ModuleDebugLineFragmentRef &Lines) {
    return Error::success();
  }

  virtual Error
  visitFileChecksums(ModuleDebugFileChecksumFragmentRef &Checksums) {
    return Error::success();
  }

  virtual Error finished() { return Error::success(); }
};

Error visitModuleDebugFragment(const ModuleDebugFragmentRecord &R,
                               ModuleDebugFragmentVisitor &V);

template <typename T>
Error visitModuleDebugFragments(T &&FragmentRange,
                                ModuleDebugFragmentVisitor &V) {
  for (const auto &L : FragmentRange) {
    if (auto EC = visitModuleDebugFragment(L, V))
      return EC;
  }
  if (auto EC = V.finished())
    return EC;
  return Error::success();
}

} // end namespace codeview

} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_MODULEDEBUGFRAGMENTVISITOR_H
