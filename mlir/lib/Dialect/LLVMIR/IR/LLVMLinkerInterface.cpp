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

#include "mlir/Linker/IRMover.h"
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

  ConflictPair join(ConflictPair existing, Operation *src) const {
    assert(!existing.hasConflict() && "expected non-conflicting pair");
    if (isLocalLinkage(getLinkage(existing.src)))
      return ConflictPair::noConflict(existing.src);
    return {existing.src, src};
  }

  ConflictPair findConflict(Operation *src) const override {
    assert(canBeLinked(src) && "expected linkable operation");

    if (isLocalLinkage(getLinkage(src)))
      return ConflictPair::noConflict(src);

    if (auto it = summary.find(getSymbol(src)); it != summary.end()) {
      return join(it->second, src);
    }

    return ConflictPair::noConflict(src);
  }

  bool isLinkNeeded(ConflictPair pair, bool forDependency) const override {
    assert(canBeLinked(pair.src) && "expected linkable operation");
    if (pair.src == pair.dst)
      return false;

    LLVM::Linkage srcLinkage = getLinkage(pair.src);

    // Always import variables with appending linkage.
    if (isAppendingLinkage(srcLinkage)) {
      return true;
    }

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

  FailureOr<bool> shouldLinkFromSource(ConflictPair pair) const {
    auto srcLinkage = getLinkage(pair.src);
    auto dstLinkage = getLinkage(pair.dst);

    auto isDeclarationForLinker = [](Operation *op) {
      if (isAvailableExternallyLinkage(getLinkage(op)))
        return true;
      return isDeclaration(op);
    };

    // Should we unconditionally use the src?
    if (shouldOverrideFromSrc())
      return true;

    // We always have to add src if it has appending linkage.
    if (isAppendingLinkage(srcLinkage) || isAppendingLinkage(dstLinkage))
      return true;

    if (isDeclarationForLinker(pair.src))
      llvm_unreachable("Not implemented");

    if (isDeclarationForLinker(pair.dst))
      return true;

    if (isCommonLinkage(srcLinkage)) {
      if (isLinkOnceLinkage(dstLinkage) || isWeakLinkage(dstLinkage))
        return true;
      if (!isCommonLinkage(dstLinkage))
        return true;
    }

    // TODO: This is not correct, should use some form of DataLayout concept
    // taking into account alignment etc
    auto srcType = pair.src->getAttrOfType<TypeAttr>("global_type");
    auto dstType = pair.dst->getAttrOfType<TypeAttr>("global_type");
    if (srcType && dstType)
      return srcType.getValue().getIntOrFloatBitWidth() >
             dstType.getValue().getIntOrFloatBitWidth();

    if (isWeakForLinker(srcLinkage)) {
      assert(!isExternalWeakLinkage(dstLinkage));
      assert(!isAvailableExternallyLinkage(dstLinkage));

      return isLinkOnceAnyLinkage(dstLinkage) || isWeakLinkage(dstLinkage);
    }

    if (isWeakForLinker(dstLinkage)) {
      assert(isExternalLinkage(srcLinkage));
      return true;
    }

    assert(!isExternalWeakLinkage(srcLinkage));
    assert(!isExternalWeakLinkage(dstLinkage));
    assert(isExternalLinkage(srcLinkage) && isExternalLinkage(dstLinkage));
    return failure();
  }

  LogicalResult resolveConflict(ConflictPair pair) override {
    assert(canBeLinked(pair.src) && "expected linkable operation");
    assert(canBeLinked(pair.dst) && "expected linkable operation");

    auto srcLinkage = getLinkage(pair.src);
    auto dstLinkage = getLinkage(pair.dst);

    auto dvar = dyn_cast<LLVM::GlobalOp>(pair.dst);
    auto svar = dyn_cast<LLVM::GlobalOp>(pair.src);
    if (!isAppendingLinkage(srcLinkage)) {
      if (dvar && svar) {
        if (isDeclaration(dvar) && isDeclaration(svar))
          if (dvar.getConstant() || svar.getConstant()) {
            dvar.setConstant(false);
            svar.setConstant(false);
          }

        if (isCommonLinkage(dstLinkage) && isCommonLinkage(srcLinkage)) {
          std::optional<int64_t> dstAlign = dvar.getAlignment();
          std::optional<int64_t> srcAlign = svar.getAlignment();
          std::optional<unsigned> align = std::nullopt;

          if (dstAlign || srcAlign)
            align = std::max(dstAlign.value_or(1), srcAlign.value_or(1));

          dvar.setAlignment(align);
          svar.setAlignment(align);
        }
      }

      // TODO: set visibility

      // TODO: set unnamed addr
    }

    FailureOr<bool> linkFromSrc = shouldLinkFromSource(pair);
    if (failed(linkFromSrc))
      return failure();

    LinkFrom comdatFrom = LinkFrom::Dst;

    if (comdatFrom == LinkFrom::Both)
      valuesToClone.insert(*linkFromSrc ? pair.dst : pair.src);

    if (*linkFromSrc)
      summary[getSymbol(pair.src)] = pair;
    return success();
  }

  void registerForLink(Operation *op) override {
    assert(canBeLinked(op) && "expected linkable operation");
    summary[getSymbol(op)] = ConflictPair::noConflict(op);
  }

  LogicalResult initialize(ModuleOp src) override {
    symbolTable = std::make_unique<SymbolTable>(src);
    return success();
  }

  LogicalResult link(ModuleOp dst) const override {
    // TODO: Implement
    // for (Operation *gv : state.getValuesToClone()) {
    //   llvm_unreachable("unimplemented");
    // }

    // TODO: Implement
    // for (unsigned i = 0, i < state.getNumValuesToLink(); ++i) {
    //   llvm_unreachable("unimplemented");
    // }

    IRMover mover(dst);
    return mover.move(summary);
  }

  Operation *prototype(ConflictPair pair) const {
    if (pair.dst)
      return pair.dst;
    return pair.src->cloneWithoutRegions();
  }

  Operation *unique(Operation *op, ModuleOp dst) const {
    SymbolTable table(dst);
    if (table.lookup(table.getSymbolName(op)))
      if (failed(table.renameToUnique(op, {})))
        return nullptr;
    return op;
  }

  Operation *materialize(ConflictPair pair, ModuleOp dst) const override {
    // Make definition if destination does not have one or has only declaration
    bool forDefinition = !pair.dst || isDeclaration(pair.dst);
    if (!forDefinition || isDeclaration(pair.src))
      return prototype(pair);

    // Definition already exists
    if (pair.dst && !isDeclaration(pair.dst))
      return pair.dst;

    assert(!isDeclaration(pair.src) && "expected source with body");
    if (!pair.dst)
      return unique(pair.src->clone(), dst);

    // TODO might need more things to clone here
    for (auto [srcRegion, dstRegion] :
         llvm::zip(pair.src->getRegions(), pair.dst->getRegions())) {
      dstRegion.takeBody(srcRegion);
    }

    return pair.dst;
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

  llvm::StringMap<ConflictPair> summary;
};

//===----------------------------------------------------------------------===//
// registerLinkerInterface
//===----------------------------------------------------------------------===//

void mlir::LLVM::registerLinkerInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    dialect->addInterfaces<LLVMSymbolLinkerInterface>();
  });
}
