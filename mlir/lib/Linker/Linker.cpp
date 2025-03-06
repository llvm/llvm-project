//===- Linker.cpp - MLIR linker implementation -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Linker/Linker.h"

#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/LinkageInterfaces.h"
#include "mlir/Linker/LinkerInterface.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"

#define DEBUG_TYPE "mlir-link-linker"

using namespace mlir;
using namespace mlir::link;
using llvm::Error;

enum class LinkFrom { Dst, Src, Both };

class ModuleLinker {
  IRMover &mover;
  OwningOpRef<Operation *> src;

  SetVector<GlobalValue> valuesToLink;

  /// For symbol clashes, prefer those from src.
  unsigned flags;

  ModuleOp getSourceModule() { return cast<ModuleOp>(src.get()); }

  bool shouldOverrideFromSrc() const { return flags & Linker::OverrideFromSrc; }
  bool shouldLinkOnlyNeeded() const { return flags & Linker::LinkOnlyNeeded; }

  bool shouldLinkFromSource(bool &linkFromSrc, GlobalValue dst,
                            GlobalValue src);

  bool linkIfNeeded(GlobalValue gv, std::vector<Operation *> &gvToClone);

  /// Given a global in the source module, return the global in the
  /// destination module that is being linked to, if any.
  GlobalValue getLinkedToGlobal(GlobalValue gv) {
    auto dst = mover.getComposite();
    // If the source has no name it can't link.  If it has local linkage,
    // there is no name match-up going on.
    if (gv.hasLocalLinkage())
      return {};

    // Otherwise see if we have a matching symbol in the destination module.
    SymbolTable dstSymbolTable(dst.getOperation());
    StringRef globalName = gv.getLinkedName();

    auto dstGlobalValueTest = dstSymbolTable.lookup(globalName);
    auto dstGlobalValue = dyn_cast_or_null<GlobalValue>(dstGlobalValueTest);
    if (!dstGlobalValue)
      return {};

    // If we found a global with the same name in the dest module, but it has
    // internal linkage, we are really not doing any linkage here.
    if (dstGlobalValue.hasLocalLinkage())
      return {};

    // Otherwise, we do in fact link to the destination global.
    return dstGlobalValue;
  }

  LogicalResult emitError(const Twine &message) {
    return src->emitError(message);
  }

public:
  ModuleLinker(IRMover &mover, OwningOpRef<Operation *> src, unsigned flags)
      : mover(mover), src(std::move(src)), flags(flags) {}
  LogicalResult run();
};

bool ModuleLinker::shouldLinkFromSource(bool &linkFromSrc, GlobalValue dst,
                                        GlobalValue src) {
  // Should we unconditionally use the src?
  if (shouldOverrideFromSrc()) {
    linkFromSrc = true;
    return false;
  }

  // We always have to add src if it has appending linkage.
  if (src.hasAppendingLinkage() || dst.hasAppendingLinkage()) {
    linkFromSrc = true;
    return false;
  }

  if (src.isDeclarationForLinker()) {
    llvm_unreachable("unimplemented");
  }

  if (dst.isDeclarationForLinker()) {
    // If dst is external but src is not:
    linkFromSrc = true;
    return false;
  }

  if (src.hasCommonLinkage()) {
    if (dst.hasLinkOnceLinkage() || dst.hasWeakLinkage()) {
      linkFromSrc = true;
      return false;
    }

    if (!dst.hasCommonLinkage()) {
      linkFromSrc = false;
      return false;
    }

    // TODO: This is not correct, should use some form of DataLayout concept
    // taking into account alignment etc
    auto srcType = src.getOperation()->getAttrOfType<TypeAttr>("global_type");
    auto dstType = dst.getOperation()->getAttrOfType<TypeAttr>("global_type");
    if (srcType && dstType) {
      linkFromSrc = srcType.getValue().getIntOrFloatBitWidth() >
                    dstType.getValue().getIntOrFloatBitWidth();
      return false;
    }
  }

  if (isWeakForLinker(src.getLinkage())) {
    assert(!dst.hasExternalWeakLinkage());
    assert(!dst.hasAvailableExternallyLinkage());

    if (dst.hasLinkOnceLinkage() && src.hasWeakLinkage()) {
      linkFromSrc = true;
      return false;
    }

    linkFromSrc = false;
    return false;
  }

  if (isWeakForLinker(dst.getLinkage())) {
    assert(src.hasExternalLinkage());
    linkFromSrc = true;
    return false;
  }

  assert(!src.hasExternalWeakLinkage());
  assert(!dst.hasExternalWeakLinkage());
  assert(dst.hasExternalLinkage() && src.hasExternalLinkage() &&
         "Unexpected linkage type!");
  return true;
  // return emitError("Linking globals named '" + src.getLinkedName() +
  //                  "': symbol multiply defined!");
}

// Returns true if no linking is needed
bool ModuleLinker::linkIfNeeded(GlobalValue gv,
                                std::vector<Operation *> &gvToClone) {
  LLVM_DEBUG(llvm::dbgs() << "ModuleLinker::linkIfNeeded<" << gv.getLinkedName()
                          << ">\n");
  GlobalValue dgv = getLinkedToGlobal(gv);

  if (shouldLinkOnlyNeeded()) {
    // Always import variables with appending linkage.
    if (!gv.hasAppendingLinkage()) {
      // Don't import globals unless they are referenced by the destination
      // module.
      if (!dgv)
        return false;
      // Don't import globals that are already defined in the destination
      // module
      if (!dgv.isDeclaration())
        return false;
    }
  }

  if (dgv && !gv.hasLocalLinkage() && !gv.hasAppendingLinkage()) {
    auto dgvar = dyn_cast<GlobalVariable>(dgv);
    auto sgvar = dyn_cast<GlobalVariable>(gv);
    if (dgvar && sgvar) {
      if (dgv.isDeclaration() && gv.isDeclaration() &&
          (!dgvar.isConstant() || !sgvar.isConstant())) {
        dgvar.setConstant(false);
        sgvar.setConstant(false);
      }

      if (dgv.hasCommonLinkage() && gv.hasCommonLinkage()) {
        auto dAlign = dgvar.getAlignment();
        auto sAlign = sgvar.getAlignment();
        std::optional<unsigned> align = std::nullopt;

        if (dAlign || sAlign)
          align = std::max(dAlign.value_or(1), sAlign.value_or(1));

        sgvar.setAlignment(align);
        dgvar.setAlignment(align);
      }
    }

    // TODO: set visibility

    // TODO: set unnamed addr
  }

  if (!dgv && !shouldOverrideFromSrc() &&
      (gv.hasLocalLinkage() || gv.hasLinkOnceLinkage() ||
       gv.hasAvailableExternallyLinkage()))
    return false;

  if (gv.isDeclaration())
    return false;

  LinkFrom comdatFrom = LinkFrom::Dst;

  bool linkFromSrc = true;
  if (dgv && shouldLinkFromSource(linkFromSrc, dgv, gv))
    return true;
  if (dgv && comdatFrom == LinkFrom::Both)
    gvToClone.push_back(linkFromSrc ? dgv.getOperation() : gv.getOperation());
  if (linkFromSrc)
    valuesToLink.insert(gv);
  return false;
}

LogicalResult ModuleLinker::run() {
  LLVM_DEBUG(llvm::dbgs() << "ModuleLinker::run" << "\n");
  auto src = getSourceModule();

  std::vector<Operation *> gvToClone;
  bool nothingToLink = true;
  src->walk([&](GlobalValue gv) {
    bool result = linkIfNeeded(gv, gvToClone);
    LLVM_DEBUG(llvm::dbgs() << "ModuleLinker::linkIfNeeded<"
                            << gv.getLinkedName() << "> = " << result << "\n");
    nothingToLink &= result;
    return WalkResult::skip();
  });

  if (nothingToLink) {
    LLVM_DEBUG(llvm::dbgs() << "Nothing to link.\n");
    return success();
  }

  // TODO: Implement
  // for (GlobalValue gv : gvToClone) {
  //   llvm_unreachable("unimplemented");
  // }

  // TODO: Implement
  // for (unsigned i = 0; i < valuesToLink.size(); ++i) {
  //   llvm_unreachable("unimplemented");
  // }

  bool hasErrors = false;
  // TODO: We are moving whatever the local src points to here (this->src), so
  // it can't be touched past this point.
  if (Error e = mover.move(std::move(this->src), valuesToLink.getArrayRef())) {
    handleAllErrors(std::move(e), [&](llvm::ErrorInfoBase &eib) {
      // TODO: Handle errors somehow
      hasErrors = true;
    });
  }

  if (hasErrors)
    return failure();

  return success();
}

LogicalResult Linker::linkInModule(OwningOpRef<ModuleOp> src) {
  unsigned flags = getFlags();

  if (!composite) {
    auto interface =
        dyn_cast_or_null<LinkerInterface>(src->getOperation()->getDialect());
    if (!interface)
      return emitError("Module does not have a linker interface");
    composite = interface->createCompositeModule(src.get());
    // We always override from source for the first module.
    flags &= Linker::OverrideFromSrc;
  }

  IRMover mover(composite.get());
  ModuleLinker modLinker(mover, std::move(src), flags);
  return modLinker.run();
}

unsigned Linker::getFlags() const {
  unsigned flags = None;

  if (config.shouldLinkOnlyNeeded())
    flags |= Linker::Flags::LinkOnlyNeeded;

  return flags;
}
