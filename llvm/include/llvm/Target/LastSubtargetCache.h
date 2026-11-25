//===- LastSubtargetCache.h - Reuse the subtarget of the last function ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Cache of the last subtarget a TargetMachine returned, to skip building the
/// subtarget key for consecutive functions with the same subtarget.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_LASTSUBTARGETCACHE_H
#define LLVM_TARGET_LASTSUBTARGETCACHE_H

#include "llvm/IR/Attributes.h"

namespace llvm {

class Module;

/// Cache of the last subtarget returned by getSubtargetImpl. A query hits if
/// the function has the same function attributes and module as the last one,
/// or else the same \p AttrsT, which must hold every input to the subtarget
/// key, including module-level ones.
///
/// Attributes are compared by their uniqued pointers, so the cache must be
/// cleared before it sees functions from a different LLVMContext.
template <typename SubtargetT, typename AttrsT> class LastSubtargetCache {
  AttributeSet FnAttrs;
  const Module *M = nullptr;
  AttrsT Attrs;
  const SubtargetT *Subtarget = nullptr;

public:
  /// Return the last subtarget if the function attributes and module match.
  const SubtargetT *lookup(AttributeSet NewFnAttrs, const Module *NewM) const {
    return Subtarget && FnAttrs == NewFnAttrs && M == NewM ? Subtarget
                                                           : nullptr;
  }

  /// Return the last subtarget if \p NewAttrs match, and remember
  /// \p NewFnAttrs and \p NewM for it.
  const SubtargetT *lookup(AttributeSet NewFnAttrs, const Module *NewM,
                           const AttrsT &NewAttrs) {
    if (!Subtarget || Attrs != NewAttrs)
      return nullptr;
    FnAttrs = NewFnAttrs;
    M = NewM;
    return Subtarget;
  }

  /// Remember \p NewSubtarget for these attributes.
  void set(AttributeSet NewFnAttrs, const Module *NewM, const AttrsT &NewAttrs,
           const SubtargetT *NewSubtarget) {
    FnAttrs = NewFnAttrs;
    M = NewM;
    Attrs = NewAttrs;
    Subtarget = NewSubtarget;
  }

  void clear() { *this = LastSubtargetCache(); }
};

} // end namespace llvm

#endif // LLVM_TARGET_LASTSUBTARGETCACHE_H
