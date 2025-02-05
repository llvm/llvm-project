//===- IRMover.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Linker/IRMover.h"
#include "llvm/Support/Error.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"

#include <assert.h>
#include <optional>

using namespace mlir;
using llvm::Error;
using llvm::Expected;

namespace {

class MLIRLinker {
  Operation *composite;
  OwningOpRef<Operation *> src;

  // TODO: This is a ValueToValueMapTy in llvm-link. I'm assuming this would be
  // the equivalent in mlir.
  IRMapping valueMap;

  DenseSet<GlobalValueLinkageOpInterface> valuesToLink;
  std::vector<GlobalValueLinkageOpInterface> worklist;
  // Replace-all-uses-with worklist
  std::vector<std::pair<Operation *, Operation *>> rauwWorklist;

  bool doneLinkingBodies;

  void maybeAdd(GlobalValueLinkageOpInterface val) {
    if (valuesToLink.insert(val).second)
      worklist.push_back(val);
  }

  std::optional<Error> foundError;
  void setError(Error e) {
    if (e)
      foundError = std::move(e);
  }

  Error linkFunctionBody(Operation *dst, FunctionLinkageOpInterface src);
  Error linkGlobalValueBody(Operation *dst, GlobalValueLinkageOpInterface src);
  bool shouldLink(Operation *dst, Operation *src);
  Operation *copyGlobalVariableProto(Operation *src, bool forDefinition);
  Operation *copyFunctionProto(Operation *src);
  Operation *copyGlobalValueProto(Operation *src, bool forIndirectSymbol);
  Expected<Operation *> linkGlobalValueProto(GlobalValueLinkageOpInterface sgv,
                                             bool forIndirectSymbol);
  void flushRAUWorklist();

  /// Given a global in the source module, return the global in the
  /// destination module that is being linked to, if any.
  GlobalValueLinkageOpInterface
  getLinkedToGlobal(GlobalValueLinkageOpInterface sgv) {

    // If the source has no name it can't link.  If it has local linkage,
    // there is no name match-up going on.
    auto symOp = dyn_cast<SymbolOpInterface>(sgv.getOperation());
    if (!symOp)
      return nullptr;

    auto sgvName = symOp.getName();
    if (sgvName.empty() || sgv.hasLocalLinkage())
      return nullptr;

    // Otherwise see if we have a match in the destination module's symtab.
    // TODO: Should the SymbolTable be a member instead?
    SymbolTable syms(composite);
    Operation *dstOp = syms.lookup(sgvName);
    if (!dstOp)
      return nullptr;

    auto dgv = dyn_cast<GlobalValueLinkageOpInterface>(dstOp);
    if (!dgv)
      return nullptr;

    // If we found a global with the same name in the dest module, but it has
    // internal linkage, we are really not doing any linkage here.
    if (dgv.hasLocalLinkage())
      return nullptr;

    // If we found an intrinsic declaration with mismatching prototypes, we
    // probably had a nameclash. Don't use that version.
    if (auto fdgv = dyn_cast<FunctionLinkageOpInterface>(sgv.getOperation())) {
      // TODO: Can we check for intrinsic functions?
    }

    // Otherwise, we do in fact link to the destination global.
    return dgv;
  }

public:
  MLIRLinker(Operation *composite, OwningOpRef<Operation *> srcOp,
             ArrayRef<GlobalValueLinkageOpInterface> valuesToLink)
      : composite{composite}, src{std::move(srcOp)} {
    for (GlobalValueLinkageOpInterface gvl : valuesToLink)
      maybeAdd(gvl);
  }
  Error run();

  Operation *materialize(GlobalValueLinkageOpInterface v,
                         bool forIndirectSymbol);
};

bool MLIRLinker::shouldLink(Operation *dst, Operation *src) {
  auto sgv = dyn_cast<GlobalValueLinkageOpInterface>(src);
  if (!sgv)
    return false;

  if (valuesToLink.count(sgv) || sgv.hasLocalLinkage()) {
    return true;
  }

  if (dst) {
    if (auto dgv = dyn_cast<GlobalValueLinkageOpInterface>(dst))
      if (!dgv.isDeclarationForLinkage())
        return false;
  }

  if (sgv.isDeclarationForLinkage()) // TODO: DoneLinkingBodies??
    return false;

  // Callback to the client to give a chance to lazily add the Global to the
  // list of value to link.
  bool LazilyAdded = false;
  //   if (AddLazyFor)
  //     AddLazyFor(SGV, [this, &LazilyAdded](GlobalValue &GV) {
  //       maybeAdd(&GV);
  //       LazilyAdded = true;
  //     });
  // TODO: Implement callback, if needed
  return LazilyAdded;
}

Operation *MLIRLinker::copyGlobalVariableProto(Operation *src,
                                               bool forDefinition) {
  // TODO: Is this the right way? Or, do we need to create an empty operation
  // and later fill in the "meat"? E.g. should we do src->cloneWithoutRegions?
  llvm_unreachable("unimplemented");
  return nullptr;
}
Operation *MLIRLinker::copyFunctionProto(Operation *src) {
  OpBuilder builder(composite->getRegion(0));

  // Clone the operation (without regions to ensure it becomes empty as is
  // considered a decl)
  Operation *newFunc = src->cloneWithoutRegions();
  builder.insert(newFunc);
  return newFunc;
}
Operation *MLIRLinker::copyGlobalValueProto(Operation *src,
                                            bool forDefinition) {
  if (auto f = dyn_cast<FunctionLinkageOpInterface>(src))
    return copyFunctionProto(src);

  // TODO: Copy metadata for global variables and function declarations?

  // TODO: If function clear personality, prefix and prologue data
  return nullptr;
}

Expected<Operation *>
MLIRLinker::linkGlobalValueProto(GlobalValueLinkageOpInterface sgv,
                                 bool forIndirectSymbol) {
  auto dgv = getLinkedToGlobal(sgv);

  bool shouldLinkOps = shouldLink(dgv.getOperation(), sgv.getOperation());

  // just missing from map
  if (shouldLinkOps) {
    if (auto existing_dst = valueMap.lookupOrNull(sgv.getOperation()))
      return existing_dst;
    // TODO: Check indicrect symbol value map, if needed.
  }

  if (!shouldLinkOps && forIndirectSymbol)
    dgv = nullptr;

  // Handle the ultra special appending linkage case first.
  // TODO: Appending linkage
  //   if (src->hasAppendingLinkage() | (dst && dst->hasAppendingLinkage())) {
  // return liAppendingVarProto()
  //   }

  bool needsRenaming = false;
  Operation *newDst;
  if (dgv && !shouldLinkOps) {
    newDst = dgv.getOperation();
  } else {
    // TODO: Done linking bodies?

    newDst = copyGlobalValueProto(sgv.getOperation(),
                                  shouldLinkOps); // TODO: || ForIndirectSymbol?
    if (shouldLinkOps) // TODO: || !ForIndirectSymbol?
      needsRenaming = true;
  }

  // TODO: overloaded intrinsics

  // if (needsRenaming)
  //   forceRenaming(newDst, srcName);

  // TODO: Comdat

  // TODO:
  // if (!shouldLinkOps && forIndirectSymbol)
  //   if (auto newDstGVL = dyn_cast<GlobalValueLinkageOpInterface>(newDst))
  //     newDstGVL.setInternalLinkage();

  // TODO: bitcasts needed??

  if (dgv && newDst != dgv.getOperation()) {
    // Schedule "replace all uses with"
    rauwWorklist.push_back(
        std::make_pair(dgv.getOperation(), newDst)); // TODO: This is simplified
  }

  return newDst;
}

Operation *MLIRLinker::materialize(GlobalValueLinkageOpInterface v,
                                   bool forIndirectSymbol) {
  Operation *op = v.getOperation();

  // If v is from dest, it was already materialized when dest was loaded.
  if (op->getParentOp() == composite)
    return nullptr;

  // When linking a global from other modules than source & dest, skip
  // materializing it because it would be mapped later when its containing
  // module is linked. Linking it now would potentially pull in many types that
  // may not be mapped properly.
  if (op->getParentOp() != src.get())
    return nullptr;

  auto newProto = linkGlobalValueProto(v, false);
  if (!newProto) {
    setError(newProto.takeError());
    return nullptr;
  }

  if (!*newProto)
    return nullptr;

  GlobalValueLinkageOpInterface newGvl =
      dyn_cast<GlobalValueLinkageOpInterface>(*newProto);
  if (!newGvl)
    return *newProto;

  // If we already created the body, just return.
  if (auto f = dyn_cast<GlobalFuncLinkageOpInterface>(newGvl.getOperation())) {
    if (!f.isDeclarationForLinkage()) {
      return *newProto;
    }
  }
  // TODO: Lots of if cases for Function, global variable, global alias.
  // for now, just check if it is a declaration, if so, not much more to do.

  // If the global is being linked for an indirect symbol, it may have already
  // been scheduled to satisfy a regular symbol. Similarly, a global being
  // linked for a regular symbol may have already been scheduled for an indirect
  // symbol. Check for these cases by looking in the other value map and
  // confirming the same value has been scheduled.  If there is an entry in the
  // ValueMap but the value is different, it means that the value already had a
  // definition in the destination module (linkonce for instance), but we need a
  // new definition for the indirect symbol ("New" will be different).
  // TODO: Some indirect symbol thing

  if (forIndirectSymbol || shouldLink(newGvl.getOperation(), v.getOperation()))
    setError(linkGlobalValueBody(newGvl.getOperation(), v));

  // TODO: Update attributes
  return newGvl.getOperation();
}

Error MLIRLinker::linkFunctionBody(Operation *dst,
                                   FunctionLinkageOpInterface src) {

  // TODO: Not exactly like this
  if (auto dgv = dyn_cast<GlobalValueLinkageOpInterface>(dst))
    assert(dgv.isDeclarationForLinkage());

  assert(!src.isDeclarationForLinkage());

  assert(src->getNumRegions() == dst->getNumRegions() &&
         "Operations must have same number of regions");

  for (auto [srcRegion, dstRegion] :
       llvm::zip(src->getRegions(), dst->getRegions())) {
    dstRegion.takeBody(srcRegion);
  }

  // TODO: several steps here, copy metadata, steal arg list and schedule
  // remapfunction. What is needed?

  // auto target = src.getOperation()->clone(mapping);
  // dst->replaceAllUsesWith(target);
  // dst is an external sym and src is not
  return Error::success();
}

Error MLIRLinker::linkGlobalValueBody(Operation *dst,
                                      GlobalValueLinkageOpInterface src) {
  // TODO: Switch on what type of thing it is. For now just assume it is a
  // function-like thing
  if (auto f = dyn_cast<FunctionLinkageOpInterface>(src.getOperation()))
    return linkFunctionBody(dst, f);

  return Error::success();
}

void MLIRLinker::flushRAUWorklist() {
  for (const auto &elem : rauwWorklist) {
    Operation *oldOp, *newOp;
    std::tie(oldOp, newOp) = elem;
    oldOp->replaceAllUsesWith(newOp);
    oldOp->erase();
  }
  rauwWorklist.clear();
}

Error MLIRLinker::run() {
  // TODO: is metadata materialization needed? Is that even a thing here?

  // TODO: Do we need to care about dbginfoformat, data layout and target
  // triple?

  // TODO: Do we need to care about CUDA?

  // TODO: Compute type mapping, if needed?

  // reverse the worklist and  process all values
  std::reverse(worklist.begin(), worklist.end());
  while (!worklist.empty()) {
    auto gvl = worklist.back();
    worklist.pop_back();

    if (valueMap.contains(
            gvl.getOperation())) // TODO: There is an indirect symbol value map,
                                 // do we need that?
      continue;

    assert(!gvl.isDeclarationForLinkage());

    // TODO: Is this the equivalent of Mapper.mapValue?
    auto newGvl = materialize(gvl, false);
    valueMap.map(gvl.getOperation(), newGvl);

    if (foundError)
      return std::move(*foundError);
    flushRAUWorklist();
  }

  // Reorder the globals just added to the destination module to match their
  // original order in the source module.
  src->walk([&](GlobalValueLinkageOpInterface gv) {
    if (gv.hasAppendingLinkage())
      return WalkResult::skip();
    if (auto op = valueMap.lookupOrNull(gv.getOperation())) {
      if (auto newValue = dyn_cast<GlobalValueLinkageOpInterface>(op)) {
        newValue->remove();
        composite->getRegion(0).back().push_back(newValue);
      }
    }

    // do not recurse into global values
    return WalkResult::skip();
  });

  return Error::success();
}
} // namespace

IRMover::IRMover(Operation *composite) : composite(composite) {}

Error IRMover::move(OwningOpRef<Operation *> src,
                    ArrayRef<GlobalValueLinkageOpInterface> valuesToLink) {

  MLIRLinker linker(composite, std::move(src), valuesToLink);
  Error e = linker.run();

  // TODO: Remove
  if (failed(verify(composite, true))) {
    llvm::outs() << "Verify failed\n";
  }
  return e;
}
