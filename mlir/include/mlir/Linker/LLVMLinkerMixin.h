//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the helper functions for the LLVM-like linkage behavior.
// It is used by the LLVMLinker and other dialects that have same linkage
// semantics.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LINKER_LLVMLINKERMIXIN_H
#define MLIR_LINKER_LLVMLINKERMIXIN_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Linker/LinkerInterface.h"

namespace mlir::link {

using Linkage = LLVM::Linkage;

//===----------------------------------------------------------------------===//
// Linkage helpers
//===----------------------------------------------------------------------===//

static inline bool isExternalLinkage(Linkage linkage) {
  return linkage == Linkage::External;
}

static inline bool isAvailableExternallyLinkage(Linkage linkage) {
  return linkage == Linkage::AvailableExternally;
}

static inline bool isLinkOnceAnyLinkage(Linkage linkage) {
  return linkage == Linkage::Linkonce;
}

static inline bool isLinkOnceODRLinkage(Linkage linkage) {
  return linkage == Linkage::LinkonceODR;
}

static inline bool isLinkOnceLinkage(Linkage linkage) {
  return isLinkOnceAnyLinkage(linkage) || isLinkOnceODRLinkage(linkage);
}

static inline bool isWeakAnyLinkage(Linkage linkage) {
  return linkage == Linkage::Weak;
}

static inline bool isWeakODRLinkage(Linkage linkage) {
  return linkage == Linkage::WeakODR;
}

static inline bool isWeakLinkage(Linkage linkage) {
  return isWeakAnyLinkage(linkage) || isWeakODRLinkage(linkage);
}

static inline bool isAppendingLinkage(Linkage linkage) {
  return linkage == Linkage::Appending;
}

static inline bool isInternalLinkage(Linkage linkage) {
  return linkage == Linkage::Internal;
}

static inline bool isPrivateLinkage(Linkage linkage) {
  return linkage == Linkage::Private;
}

static inline bool isLocalLinkage(Linkage linkage) {
  return isInternalLinkage(linkage) || isPrivateLinkage(linkage);
}

static inline bool isExternalWeakLinkage(Linkage linkage) {
  return linkage == Linkage::ExternWeak;
}

static inline bool isCommonLinkage(Linkage linkage) {
  return linkage == Linkage::Common;
}

static inline bool isWeakForLinker(Linkage linkage) {
  return linkage == Linkage::Weak || linkage == Linkage::WeakODR ||
         linkage == Linkage::Linkonce || linkage == Linkage::LinkonceODR ||
         linkage == Linkage::Common || linkage == Linkage::ExternWeak;
}

//===----------------------------------------------------------------------===//
// Visibility helpers
//===----------------------------------------------------------------------===//

using Visibility = LLVM::Visibility;

static inline bool isHiddenVisibility(Visibility visibility) {
  return visibility == Visibility::Hidden;
}

static inline bool isProtectedVisibility(Visibility visibility) {
  return visibility == Visibility::Protected;
}

static inline Visibility getMinVisibility(Visibility lhs, Visibility rhs) {
  if (isHiddenVisibility(lhs) || isHiddenVisibility(rhs))
    return Visibility::Hidden;
  if (isProtectedVisibility(lhs) || isProtectedVisibility(rhs))
    return Visibility::Protected;
  return Visibility::Default;
}

//===----------------------------------------------------------------------===//
// Unnamed_addr helpers
//===----------------------------------------------------------------------===//

using UnnamedAddr = LLVM::UnnamedAddr;

static bool isNoneUnnamedAddr(UnnamedAddr val) {
  return val == UnnamedAddr::None;
}

static bool isLocalUnnamedAddr(UnnamedAddr val) {
  return val == UnnamedAddr::Local;
}

static UnnamedAddr getMinUnnamedAddr(UnnamedAddr lhs, UnnamedAddr rhs) {
  if (isNoneUnnamedAddr(lhs) || isNoneUnnamedAddr(rhs))
    return UnnamedAddr::None;
  if (isLocalUnnamedAddr(lhs) || isLocalUnnamedAddr(rhs))
    return UnnamedAddr::Local;
  return UnnamedAddr::Global;
}

//===----------------------------------------------------------------------===//
// LLVMLinkerMixin
//===----------------------------------------------------------------------===//

enum class ConflictResolution {
  LinkFromSrc,
  LinkFromDst,
  LinkFromBothAndRenameDst,
  LinkFromBothAndRenameSrc,
};

template <typename DerivedLinkerInterface>
class LLVMLinkerMixin {
  const DerivedLinkerInterface &getDerived() const {
    return static_cast<const DerivedLinkerInterface &>(*this);
  }

public:
  bool isDeclarationForLinker(Operation *op) const {
    const DerivedLinkerInterface &derived = getDerived();
    if (isAvailableExternallyLinkage(derived.getLinkage(op)))
      return true;
    return derived.isDeclaration(op);
  }

  bool isLinkNeeded(Conflict pair, bool forDependency) const {
    const DerivedLinkerInterface &derived = getDerived();
    assert(derived.canBeLinked(pair.src) && "expected linkable operation");
    if (pair.src == pair.dst)
      return false;

    Linkage srcLinkage = derived.getLinkage(pair.src);

    // Always import variables with appending linkage.
    if (isAppendingLinkage(srcLinkage))
      return true;

    bool alreadyDeclared = pair.dst && derived.isDeclaration(pair.dst);

    // Don't import globals that are already declared
    if (derived.shouldLinkOnlyNeeded() && !alreadyDeclared)
      return false;

    // Private dependencies are gonna be renamed and linked
    if (isLocalLinkage(srcLinkage))
      return forDependency;

    // Always import dependencies that are not yet defined or declared
    if (forDependency && !pair.dst)
      return true;

    if (derived.isDeclaration(pair.src))
      return false;

    if (derived.shouldOverrideFromSrc())
      return true;

    if (pair.dst)
      return true;

    // Linkage specifies to keep operation only in source
    return !(isLinkOnceLinkage(srcLinkage) ||
             isAvailableExternallyLinkage(srcLinkage));
  }

  LogicalResult verifyLinkageCompatibility(Conflict pair) {
    const DerivedLinkerInterface &derived = getDerived();
    assert(derived.canBeLinked(pair.src) && "expected linkable operation");
    assert(derived.canBeLinked(pair.dst) && "expected linkable operation");

    auto linkError = [&](const Twine &error) -> LogicalResult {
        return pair.src->emitError(error) << " dst: " << pair.dst->getLoc();
    };

    Linkage srcLinkage = derived.getLinkage(pair.src);
    Linkage dstLinkage = derived.getLinkage(pair.dst);

    UnnamedAddr srcUnnamedAddr = derived.getUnnamedAddr(pair.src);
    UnnamedAddr dstUnnamedAddr = derived.getUnnamedAddr(pair.dst);

    if (isAppendingLinkage(srcLinkage) && isAppendingLinkage(dstLinkage)) {
      if (srcUnnamedAddr != dstUnnamedAddr) {
        return linkError("Appending variables with different unnamed_addr need to be linked");
      }
    }
    return success();
  }

  ConflictResolution resolveConflict(Conflict pair) {
    const DerivedLinkerInterface &derived = getDerived();
    assert(derived.canBeLinked(pair.src) && "expected linkable operation");
    assert(derived.canBeLinked(pair.dst) && "expected linkable operation");

    Linkage srcLinkage = derived.getLinkage(pair.src);
    Linkage dstLinkage = derived.getLinkage(pair.dst);

    Visibility srcVisibility = derived.getVisibility(pair.src);
    Visibility dstVisibility = derived.getVisibility(pair.dst);
    Visibility visibility = getMinVisibility(srcVisibility, dstVisibility);

    derived.setVisibility(pair.src, visibility);
    derived.setVisibility(pair.dst, visibility);

    UnnamedAddr srcUnnamedAddr = derived.getUnnamedAddr(pair.src);
    UnnamedAddr dstUnnamedAddr = derived.getUnnamedAddr(pair.dst);

    UnnamedAddr unnamedAddr = getMinUnnamedAddr(srcUnnamedAddr, dstUnnamedAddr);
    derived.setUnnamedAddr(pair.src, unnamedAddr);
    derived.setUnnamedAddr(pair.dst, unnamedAddr);

    const bool srcIsDeclaration = isDeclarationForLinker(pair.src);
    const bool dstIsDeclaration = isDeclarationForLinker(pair.dst);

    if (isAvailableExternallyLinkage(srcLinkage) && dstIsDeclaration) {
      return ConflictResolution::LinkFromSrc;
    }

    // If both `src` and `dst` are declarations, we can ignore the conflict
    // and keep the `dst` declaration.
    if (srcIsDeclaration && dstIsDeclaration)
      return ConflictResolution::LinkFromDst;

    // If the `dst` is a declaration import `src` definition
    // Link an available_externally over a declaration.
    if (dstIsDeclaration && !srcIsDeclaration)
      return ConflictResolution::LinkFromSrc;

    // Conflicting private values are to be renamed.
    if (isLocalLinkage(dstLinkage))
      return ConflictResolution::LinkFromBothAndRenameDst;

    if (isLocalLinkage(srcLinkage))
      return ConflictResolution::LinkFromBothAndRenameSrc;

    if (isLinkOnceLinkage(srcLinkage))
      return ConflictResolution::LinkFromDst;

    if (isLinkOnceLinkage(dstLinkage) || isWeakLinkage(dstLinkage))
      return ConflictResolution::LinkFromSrc;

    if (isCommonLinkage(srcLinkage)) {
      if (!isCommonLinkage(dstLinkage))
        return ConflictResolution::LinkFromDst;
      if (derived.getBitWidth(pair.src) > derived.getBitWidth(pair.dst))
        return ConflictResolution::LinkFromSrc;
      return ConflictResolution::LinkFromDst;
    }

    if (isWeakForLinker(srcLinkage)) {
      assert(!isExternalWeakLinkage(dstLinkage));
      assert(!isAvailableExternallyLinkage(dstLinkage));
      if (isLinkOnceLinkage(dstLinkage) && isWeakLinkage(srcLinkage)) {
        return ConflictResolution::LinkFromSrc;
      } else {
        // No need to link the `src`
        return ConflictResolution::LinkFromDst;
      }
    }

    if (isWeakForLinker(dstLinkage)) {
      assert(isExternalLinkage(srcLinkage));
      return ConflictResolution::LinkFromSrc;
    }

    llvm_unreachable("unimplemented conflict resolution");
  }
};

//===----------------------------------------------------------------------===//
// SymbolAttrLLVMLinkerMixin
//===----------------------------------------------------------------------===//

template <typename DerivedLinkerInterface>
class SymbolAttrLLVMLinkerInterface
    : public SymbolAttrLinkerInterface,
      public LLVMLinkerMixin<DerivedLinkerInterface> {
public:
  using SymbolAttrLinkerInterface::SymbolAttrLinkerInterface;

  using LinkerMixin = LLVMLinkerMixin<DerivedLinkerInterface>;

  bool isLinkNeeded(Conflict pair, bool forDependency) const override {
    return LinkerMixin::isLinkNeeded(pair, forDependency);
  }

  LogicalResult resolveConflict(Conflict pair) override {
    if (failed(LinkerMixin::verifyLinkageCompatibility(pair)))
        return failure();
    ConflictResolution resolution = LinkerMixin::resolveConflict(pair);

    switch (resolution) {
    case ConflictResolution::LinkFromSrc:
      registerForLink(pair.src);
      return success();
    case ConflictResolution::LinkFromDst:
      return success();
    case ConflictResolution::LinkFromBothAndRenameDst:
      uniqued.insert(pair.dst);
      registerForLink(pair.src);
      return success();
    case ConflictResolution::LinkFromBothAndRenameSrc:
      uniqued.insert(pair.src);
      return success();
    }

    llvm_unreachable("unimplemented conflict resolution");
  }
};

} // namespace mlir::link

#endif // MLIR_LINKER_LLVMLINKERMIXIN_H
