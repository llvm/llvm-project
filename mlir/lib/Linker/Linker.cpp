//===- Linker.cpp - MLIR linker implementation -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Linker/Linker.h"

#include "mlir/Interfaces/LinkageInterfaces.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace mlir::link;

enum class LinkFrom { Dst, Src, Both };

class ModuleLinker {
  IRMover &mover;
  OwningOpRef<Operation *> src;

  SetVector<Operation *> valuesToLink;

  /// For symbol clashes, prefer those from src.
  unsigned flags;

  /// List of global value names that should be internalized.
  StringSet<> internalize;

  /// Function that will perform the actual internalization. The reason for a
  /// callback is that the linker cannot call internalizeModule without
  /// creating a circular dependency between IPO and the linker.
  InternalizeCallbackFn internalizeCallback;

  LinkableModuleOpInterface getSourceModule() {
    return cast<LinkableModuleOpInterface>(src.get());
  }

  DenseMap<StringRef, std::pair<ComdatSelectionKind, LinkFrom>> comdatsChosen;

  bool getComdatResult(StringRef srcSymbol,
                       ComdatSelectionKind srcSelectionKind,
                       ComdatSelectionKind &resultSelectionKind,
                       LinkFrom &from);

  bool computeResultingSelectionKind(StringRef comdatName,
                                     ComdatSelectionKind src,
                                     ComdatSelectionKind dst,
                                     ComdatSelectionKind &result,
                                     LinkFrom &from);

  /// Drop GV if it is a member of a comdat that we are dropping.
  /// This can happen with COFF's largest selection kind.
  void dropReplacedComdat(GlobalValueLinkageOpInterface gv,
                          DenseSet<Operation *> &replacedDstComdats);

  bool linkIfNeeded(GlobalValueLinkageOpInterface gv,
                    std::vector<Operation *> &gvToClone);

public:
  ModuleLinker(IRMover &mover, OwningOpRef<Operation *> src, unsigned flags,
               InternalizeCallbackFn internalizeCallback = {})
      : mover(mover), src(std::move(src)), flags(flags),
        internalizeCallback(std::move(internalizeCallback)) {}
  LogicalResult run();
};

bool ModuleLinker::getComdatResult(StringRef srcSymbol,
                                   ComdatSelectionKind srcSelectionKind,
                                   ComdatSelectionKind &resultSelectionKind,
                                   LinkFrom &from) {
  auto dst = mover.getComposite();
  // TODO: Compute once and pass reference along.
  auto dstComdatSymTab = dst.getComdatSymbolTable();

  if (auto it = comdatsChosen.find(srcSymbol); it == comdatsChosen.end()) {
    // Use the comdat if it is only available in one of the modules.
    from = LinkFrom::Src;
    resultSelectionKind = srcSelectionKind;
    return false;
  } else {
    auto [dstComdat, dstSelectionKind] = *it;
    return computeResultingSelectionKind(srcSymbol, srcSelectionKind,
                                         dstSelectionKind.first,
                                         resultSelectionKind, from);
  }
}

static bool isAnyOrLargest(ComdatSelectionKind kind) {
  return kind == ComdatSelectionKind::Any ||
         kind == ComdatSelectionKind::Largest;
}

static bool isLargest(ComdatSelectionKind kind) {
  return kind == ComdatSelectionKind::Largest;
}

bool ModuleLinker::computeResultingSelectionKind(StringRef comdatName,
                                                 ComdatSelectionKind src,
                                                 ComdatSelectionKind dst,
                                                 ComdatSelectionKind &result,
                                                 LinkFrom &from) {
  // The ability to mix Comdat::SelectionKind::Any with
  // Comdat::SelectionKind::Largest is a behavior that comes from COFF.
  bool dstAnyOrLargest = isAnyOrLargest(dst);
  bool srcAnyOrLargest = isAnyOrLargest(src);

  if (dstAnyOrLargest && srcAnyOrLargest) {
    if (isLargest(dst) || isLargest(src))
      result = ComdatSelectionKind::Largest;
    else
      result = ComdatSelectionKind::Any;
  } else if (src == dst) {
    result = dst;
  } else {
    // TODO: emitError("Linking COMDATs named '" + comdatName + "': invalid
    // selection kinds!");
  }

  switch (result) {
  case ComdatSelectionKind::Any:
    from = LinkFrom::Dst;
    break;
  case ComdatSelectionKind::NoDeduplicate:
    from = LinkFrom::Both;
    break;
  case ComdatSelectionKind::ExactMatch:
  case ComdatSelectionKind::Largest:
  case ComdatSelectionKind::SameSize:
    llvm_unreachable("unimplemented");
  }

  return false;
}

void ModuleLinker::dropReplacedComdat(
    GlobalValueLinkageOpInterface gv,
    DenseSet<Operation *> &replacedDstComdats) {
  auto comdat = gv.getComdatSelectionKind();
  if (!comdat)
    return;
  if (!replacedDstComdats.count(gv.getOperation()))
    return;

  llvm_unreachable("unimplemented");
}

bool ModuleLinker::linkIfNeeded(GlobalValueLinkageOpInterface gv,
                                std::vector<Operation *> &gvToClone) {
  return false;
}

LogicalResult ModuleLinker::run() {
  auto dst = mover.getComposite();
  auto src = getSourceModule();
  DenseSet<Operation *> replacedDstComdats;
  DenseSet<Operation *> nonPrevailingComdats;

  for (auto &[symbol, comdat] : src.getComdatSymbolTable()) {
    if (comdatsChosen.count(symbol))
      continue;

    llvm_unreachable("unimplemented");
  }

  // TODO add `globals` and other values to LinkableModuleOpInterface
  dst->walk([&](GlobalValueLinkageOpInterface gv) {
    dropReplacedComdat(gv, replacedDstComdats);
    // do not recurse into global values
    return WalkResult::skip();
  });

  if (!nonPrevailingComdats.empty()) {
    llvm_unreachable("unimplemented");
  }

  // TODO add `globals` and other values to LinkableModuleOpInterface
  src->walk([&](GlobalValueLinkageOpInterface gv) {
    if (gv.hasLinkOnceLinkage()) {
      if (auto comdat = gv.getComdatSelectionKind()) {
        llvm_unreachable("unimplemented lazy comdat members");
      }
    }
    // do not recurse into global values
    return WalkResult::skip();
  });

  std::vector<Operation *> gvToClone;
  bool anythingToLink = false;
  src->walk([&](GlobalValueLinkageOpInterface gv) {
    anythingToLink |= linkIfNeeded(gv, gvToClone);
    return WalkResult::skip();
  });

  if (!anythingToLink)
    return success();

  for ([[maybe_unused]] auto gv : gvToClone) {
    llvm_unreachable("unimplemented");
  }

  for (unsigned i = 0; i < valuesToLink.size(); ++i) {
    llvm_unreachable("unimplemented");
  }

  if (internalizeCallback) {
    for (auto gv : valuesToLink) {
      internalize.insert(cast<SymbolOpInterface>(gv).getName());
    }
  }

  // TODO deal with errors

  if (internalizeCallback) {
    internalizeCallback(dst, internalize);
  }

  return success();
}

Linker::Linker(LinkableModuleOpInterface composite) : mover(composite) {}

LogicalResult Linker::linkInModule(OwningOpRef<Operation *> src, unsigned flags,
                                   InternalizeCallbackFn internalizeCallback) {
  ModuleLinker modLinker(mover, std::move(src), flags,
                         std::move(internalizeCallback));
  return modLinker.run();
}
