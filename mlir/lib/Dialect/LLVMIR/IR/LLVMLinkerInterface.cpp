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
// LLVMLinkerState
//===----------------------------------------------------------------------===//

struct LLVMLinkerState : LinkerState::Base<LLVMLinkerState> {
  LLVMLinkerState(ModuleOp src) : symbolTable(src) {}

  static std::unique_ptr<LinkerState> create(ModuleOp src) {
    return std::make_unique<LLVMLinkerState>(src);
  }

  Operation *lookup(Operation *op) const override {
    return symbolTable.lookup(symbolTable.getSymbolName(op));
  }

  void pushValuesToLink(ConflictPair conflict) {
    valuesToLink.insert(conflict);
  }

  void pushValueToClone(Operation *op) { valuesToClone.insert(op); }

  ArrayRef<ConflictPair> getValuesToLink() const {
    return valuesToLink.getArrayRef();
  }

  unsigned getNumValuesToLink() const { return valuesToLink.size(); }

  FailureOr<StringAttr> rename(Operation *op) {
    return symbolTable.renameToUnique(op, {});
  }

  ArrayRef<Operation *> getValuesToClone() const {
    return valuesToClone.getArrayRef();
  }

private:
  SymbolTable symbolTable;
  SetVector<Operation *> valuesToClone;
  SetVector<ConflictPair> valuesToLink;
};

//===----------------------------------------------------------------------===//
// LLVMSymbolLinkerInterface
//===----------------------------------------------------------------------===//

struct LLVMSymbolLinkerInterface
    : SymbolLinkerInterface::Base<LLVMSymbolLinkerInterface, LLVMLinkerState> {

  using Base =
      SymbolLinkerInterface::Base<LLVMSymbolLinkerInterface, LLVMLinkerState>;

  LLVMSymbolLinkerInterface(Dialect *dialect) : Base(dialect) {}

  enum class LinkFrom { Dst, Src, Both };

  ModuleLinkerInterface *getModuleLinkerInterface(Operation *op) const {
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    if (!mod)
      return nullptr;
    return dyn_cast_or_null<ModuleLinkerInterface>(mod->getDialect());
  }

  bool canBeLinked(Operation *op) const override {
    return isa<LLVM::GlobalOp>(op) || isa<LLVM::LLVMFuncOp>(op);
  }

  StringRef getSymbol(Operation *op) const override { return symbol(op); }

  ConflictPair findConflict(Operation *src) const override {
    assert(canBeLinked(src) && "expected linkable operation");

    if (isLocalLinkage(getLinkage(src)))
      return {nullptr, src};

    // TODO: make lookup through module state
    if (ModuleLinkerInterface *iface = getModuleLinkerInterface(src)) {
      if (Operation *dst = lookup(src)) {
        if (dst != src && !isLocalLinkage(getLinkage(dst))) {
          return {dst, src};
        }
      }
    }

    return {nullptr, src};
  }

  bool isLinkNeeded(ConflictPair pair) const override {
    assert(canBeLinked(pair.src) && "expected linkable operation");

    LLVM::Linkage srcLinkage = getLinkage(pair.src);

    // Always import variables with appending linkage.
    if (isAppendingLinkage(srcLinkage)) {
      return true;
    }

    if (shouldLinkOnlyNeeded()) {
      // Don't import globals that are already defined
      if (!pair.dst || !isDeclaration(pair.dst))
        return false;
    }
    if (isDeclaration(pair.src))
      return false;

    bool keepOnlyInSource = isLocalLinkage(srcLinkage) ||
                            isLinkOnceLinkage(srcLinkage) ||
                            isAvailableExternallyLinkage(srcLinkage);

    return pair.dst || shouldOverrideFromSrc() || !keepOnlyInSource;
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

    LLVMLinkerState &state = getLinkerState();
    if (comdatFrom == LinkFrom::Both)
      state.pushValueToClone(*linkFromSrc ? pair.dst : pair.src);
    if (*linkFromSrc)
      state.pushValuesToLink(pair);

    return success();
  }

  void registerOperation(Operation *op) override {
    assert(canBeLinked(op) && "expected linkable operation");
    getLinkerState().pushValuesToLink({{}, op});
  }

  std::unique_ptr<LinkerState> init(ModuleOp src) const override {
    return LLVMLinkerState::create(src);
  }

  LogicalResult link(ModuleOp dst) const override {
    const LLVMLinkerState &state = getLinkerState();

    // TODO: Implement
    // for (Operation *gv : state.getValuesToClone()) {
    //   llvm_unreachable("unimplemented");
    // }

    // TODO: Implement
    // for (unsigned i = 0, i < state.getNumValuesToLink(); ++i) {
    //   llvm_unreachable("unimplemented");
    // }

    // move values
    IRMover mover(dst);
    return mover.move(state.getValuesToLink());
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
    if (!forDefinition)
      return prototype(pair);

    // Definition laready exists
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
};

//===----------------------------------------------------------------------===//
// registerLinkerInterface
//===----------------------------------------------------------------------===//

void mlir::LLVM::registerLinkerInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    dialect->addInterfaces<LLVMSymbolLinkerInterface>();
  });
}
