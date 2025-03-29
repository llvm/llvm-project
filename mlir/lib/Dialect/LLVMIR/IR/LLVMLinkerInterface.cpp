//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to link llvm dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Linker/LinkerInterface.h"

using namespace mlir;
using namespace mlir::link;

using Linkage = LLVM::Linkage;

static Linkage getLinkage(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getLinkage();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.getLinkage();
  llvm_unreachable("unexpected operation");
}

static bool isExternalLinkage(Linkage linkage) {
  return linkage == Linkage::External;
}

static bool isAvailableExternallyLinkage(Linkage linkage) {
  return linkage == Linkage::AvailableExternally;
}

static bool isLinkOnceAnyLinkage(Linkage linkage) {
  return linkage == Linkage::Linkonce;
}

static bool isLinkOnceODRLinkage(Linkage linkage) {
  return linkage == Linkage::LinkonceODR;
}

static bool isLinkOnceLinkage(Linkage linkage) {
  return isLinkOnceAnyLinkage(linkage) || isLinkOnceODRLinkage(linkage);
}

static bool isWeakAnyLinkage(Linkage linkage) {
  return linkage == Linkage::Weak;
}

static bool isWeakODRLinkage(Linkage linkage) {
  return linkage == Linkage::WeakODR;
}

static bool isWeakLinkage(Linkage linkage) {
  return isWeakAnyLinkage(linkage) || isWeakODRLinkage(linkage);
}

LLVM_ATTRIBUTE_UNUSED static bool isAppendingLinkage(Linkage linkage) {
  return linkage == Linkage::Appending;
}

static bool isInternalLinkage(Linkage linkage) {
  return linkage == Linkage::Internal;
}

static bool isPrivateLinkage(Linkage linkage) {
  return linkage == Linkage::Private;
}

static bool isLocalLinkage(Linkage linkage) {
  return isInternalLinkage(linkage) || isPrivateLinkage(linkage);
}

static bool isExternalWeakLinkage(Linkage linkage) {
  return linkage == Linkage::ExternWeak;
}

LLVM_ATTRIBUTE_UNUSED static bool isCommonLinkage(Linkage linkage) {
  return linkage == Linkage::Common;
}

LLVM_ATTRIBUTE_UNUSED static bool isValidDeclarationLinkage(Linkage linkage) {
  return isExternalWeakLinkage(linkage) || isExternalLinkage(linkage);
}

/// Whether the definition of this global may be replaced by something
/// non-equivalent at link time. For example, if a function has weak linkage
/// then the code defining it may be replaced by different code.
LLVM_ATTRIBUTE_UNUSED static bool isInterposableLinkage(Linkage linkage) {
  switch (linkage) {
  case Linkage::Weak:
  case Linkage::Linkonce:
  case Linkage::Common:
  case Linkage::ExternWeak:
    return true;

  case Linkage::AvailableExternally:
  case Linkage::LinkonceODR:
  case Linkage::WeakODR:
    // The above three cannot be overridden but can be de-refined.

  case Linkage::External:
  case Linkage::Appending:
  case Linkage::Internal:
  case Linkage::Private:
    return false;
  }
  llvm_unreachable("Fully covered switch above!");
}

/// Whether the definition of this global may be discarded if it is not used
/// in its compilation unit.
LLVM_ATTRIBUTE_UNUSED static bool isDiscardableIfUnused(Linkage linkage) {
  return isLinkOnceLinkage(linkage) || isLocalLinkage(linkage) ||
         isAvailableExternallyLinkage(linkage);
}

/// Whether the definition of this global may be replaced at link time.  NB:
/// Using this method outside of the code generators is almost always a
/// mistake: when working at the IR level use isInterposable instead as it
/// knows about ODR semantics.
LLVM_ATTRIBUTE_UNUSED static bool isWeakForLinker(Linkage linkage) {
  return linkage == Linkage::Weak || linkage == Linkage::WeakODR ||
         linkage == Linkage::Linkonce || linkage == Linkage::LinkonceODR ||
         linkage == Linkage::Common || linkage == Linkage::ExternWeak;
}

LLVM_ATTRIBUTE_UNUSED static bool isValidLinkage(Linkage linkage) {
  return isExternalLinkage(linkage) || isLocalLinkage(linkage) ||
         isWeakLinkage(linkage) || isLinkOnceLinkage(linkage);
}

StringRef symbol(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getSymName();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.getSymName();
  llvm_unreachable("unexpected operation");
}

bool isDeclaration(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getInitializerRegion().empty() && !gv.getValue();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.getBody().empty();
  llvm_unreachable("unexpected operation");
}

//===----------------------------------------------------------------------===//
// LLVMSymbolLinkerInterface
//===----------------------------------------------------------------------===//

class LLVMSymbolLinkerInterface : public SymbolLinkerInterface {
public:
  using SymbolLinkerInterface::SymbolLinkerInterface;

  enum class LinkFrom { Dst, Src, Both };

  bool canBeLinked(Operation *op) const override {
    return isa<LLVM::GlobalOp>(op) || isa<LLVM::LLVMFuncOp>(op);
  }

  StringRef getSymbol(Operation *op) const override { return symbol(op); }

  Conflict findConflict(Operation *src) const override {
    assert(canBeLinked(src) && "expected linkable operation");

    if (auto it = summary.find(getSymbol(src)); it != summary.end()) {
      return {it->second, src};
    }

    return Conflict::noConflict(src);
  }

  bool isLinkNeeded(Conflict pair, bool forDependency) const override {
    assert(canBeLinked(pair.src) && "expected linkable operation");
    if (pair.src == pair.dst)
      return false;

    LLVM::Linkage srcLinkage = getLinkage(pair.src);

    // Always import variables with appending linkage.
    if (isAppendingLinkage(srcLinkage))
      return true;

    bool alreadyDeclared = pair.dst && isDeclaration(pair.dst);

    // Don't import globals that are already defined
    if (shouldLinkOnlyNeeded() && !alreadyDeclared)
      return false;

    // Always import dependencies that are not yet defined or declared
    if (forDependency && !pair.dst)
      return true;

    if (isDeclaration(pair.src))
      return false;

    if (pair.hasConflict())
      return true;

    if (shouldOverrideFromSrc())
      return true;

    // linkage specifies to keep operation only in source
    return !(isLocalLinkage(srcLinkage) || isLinkOnceLinkage(srcLinkage) ||
             isAvailableExternallyLinkage(srcLinkage));
  }

  LogicalResult resolveConflict(Conflict pair) override {
    assert(canBeLinked(pair.src) && "expected linkable operation");
    assert(canBeLinked(pair.dst) && "expected linkable operation");

    // If both `src` and `dst` are declarations, we can ignore the conflict.
    if (isDeclaration(pair.src) && isDeclaration(pair.dst)) {
      return success();
    }

    // If the `dst` is a declaration import `src` definition
    if (isDeclaration(pair.dst) && !isDeclaration(pair.src)) {
      registerForLink(pair.src);
      return success();
    }

    llvm_unreachable("unimplemented conflict resolution");
  }

  void registerForLink(Operation *op) override {
    assert(canBeLinked(op) && "expected linkable operation");
    summary[getSymbol(op)] = op;
  }

  LogicalResult initialize(ModuleOp src) override {
    symbolTable = std::make_unique<SymbolTable>(src);
    return success();
  }

  LogicalResult link(LinkState &state) const override {
    for (const auto &[symbol, op] : summary) {
      if (!materialize(op, state)) {
        return failure();
      }
    }

    return success();
  }

  Operation *materialize(Operation *src, LinkState &state) const override {
    return state.clone(src);
  }

  SmallVector<Operation *> dependencies(Operation *op) const override {
    SmallVector<Operation *> result;

    Operation *symbolTableOp = symbolTable->getOp();
    op->walk([&](SymbolUserOpInterface user) {
      if (user.getOperation() == op)
        return;

      if (SymbolRefAttr symbol = user.getUserSymbol())
        if (Operation *dep = symbolTable->lookupSymbolIn(symbolTableOp, symbol))
          result.push_back(dep);
    });

    return result;
  }

private:
  std::unique_ptr<SymbolTable> symbolTable;

  SetVector<Operation *> valuesToClone;

  llvm::StringMap<Operation *> summary;
};

//===----------------------------------------------------------------------===//
// registerLinkerInterface
//===----------------------------------------------------------------------===//

void mlir::LLVM::registerLinkerInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    dialect->addInterfaces<LLVMSymbolLinkerInterface>();
  });
}
