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

namespace {

class MLIRLinker {
  Operation *composite;
  OwningOpRef<Operation *> src;

  // TODO: This is a ValueToValueMapTy in llvm-link. I'm assuming this would be
  // the equivalent in mlir.
  IRMapping valueMap;

  DenseSet<Operation *> valuesToLink;
  std::vector<Operation *> worklist;
  // Replace-all-uses-with worklist
  std::vector<std::pair<Operation *, Operation *>> rauwWorklist;

  void maybeAdd(Operation *val) {
    if (valuesToLink.insert(val).second)
      worklist.push_back(val);
  }

  std::optional<Error> foundError;
  void setError(Error e) {
    if (e)
      foundError = std::move(e);
  }

  Error linkFunctionBody(Operation *dst, FunctionLinkageOpInterface src);
  Error linkGlobalValueBody(Operation *dst, Operation *src);
  bool shouldLink(Operation *dst, Operation *src);
  Operation *copyGlobalVariableProto(Operation *src, bool forDefinition);
  Operation *copyFunctionProto(Operation *src);
  Operation *copyGlobalValueProto(Operation *src, bool forIndirectSymbol);
  Operation *linkGlobalValueProto(Operation *src, bool forIndirectSymbol);
  void flushRAUWorklist();

  // TODO: Consider sharing with linker
  std::optional<StringRef> getSymbol(Operation *op) {
    if (auto symbolOp = dyn_cast<SymbolOpInterface>(op)) {
      return {symbolOp.getName()};
    }
    return {};
  }

  // TODO: Probably rename to something like getLinkedToOperation
  // TODO: COnsider if it should be shared with the linker instead of copying
  // it.
  Operation *getLinkedToGlobal(Operation *srcOp) {

    // If the source has no name it can't link.  If it has local linkage,
    // there is no name match-up going on.
    auto SrcSymOpt = getSymbol(srcOp);
    if (!SrcSymOpt)
      return nullptr;

    auto SrcSym = *SrcSymOpt;

    if (auto srcGVL = dyn_cast<GlobalValueLinkageOpInterface>(srcOp)) {
      if (srcGVL.hasLocalLinkage())
        return nullptr;
    }

    // Is there a match in the destination symbol table?
    // TODO: Should the SymbolTable be a member instead?
    SymbolTable syms(composite);
    Operation *DstOp = syms.lookup(SrcSym);

    if (!DstOp)
      return nullptr;

    // If the dst-operation has local linkage, it shouldn't be linked
    if (auto dstGVL = dyn_cast<GlobalValueLinkageOpInterface>(DstOp)) {
      if (dstGVL.hasLocalLinkage())
        return nullptr;

      // TODO: Do a check on instrinsic function mismatch?
      return dstGVL;
    }

    return nullptr;
  }

public:
  MLIRLinker(Operation *composite, OwningOpRef<Operation *> srcOp,
             ArrayRef<Operation *> valuesToLink)
      : composite{composite}, src{std::move(srcOp)} {
    for (Operation *op : valuesToLink)
      maybeAdd(op);
  }
  Error run();

  Operation *materialize(Operation *op);
};

bool MLIRLinker::shouldLink(Operation *dst, Operation *src) {
  auto srcGVL = dyn_cast<GlobalValueLinkageOpInterface>(src);
  if (!srcGVL)
    return false;

  if (valuesToLink.count(src) || srcGVL.hasLocalLinkage()) {
    return true;
  }

  if (dst) {
    if (auto dstGVL = dyn_cast<GlobalValueLinkageOpInterface>(dst))
      if (!dstGVL.isDeclarationForLinkage())
        return false;
  }

  if (srcGVL.isDeclarationForLinkage()) // TODO: DoneLinkingBodies??
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

Operation *MLIRLinker::linkGlobalValueProto(Operation *src,
                                            bool forIndirectSymbol) {
  auto dst = getLinkedToGlobal(src);
  bool shouldLinkOps = shouldLink(dst, src);
  if (shouldLinkOps) {
    if (auto existing_dst = valueMap.lookupOrNull(src))
      return existing_dst;
    // TODO: Check indicrect symbol value map, if needed.
  }

  if (!shouldLinkOps && forIndirectSymbol)
    dst = nullptr;

  // TODO: Appending linkage
  // special case ,appending linkage
  //   if (src->hasAppendingLinkage() | (dst && dst->hasAppendingLinkage())) {
  // return liAppendingVarProto()
  //   }

  bool needsRenaming = false;
  Operation *newDst;
  if (dst && !shouldLinkOps) {
    newDst = dst;
  } else {
    // TODO: Done linking bodies?

    newDst =
        copyGlobalValueProto(src, shouldLinkOps); // TODO: || ForIndirectSymbol?
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

  if (dst && newDst != dst) {
    // Schedule "replace all uses with"
    rauwWorklist.push_back(
        std::make_pair(dst, newDst)); // TODO: This is simplified
  }

  return newDst;
}

Operation *MLIRLinker::materialize(Operation *op) {
  // TODO: This is an argument to the materialize in llvm-link
  bool forIndirectSymbol = false;

  // If the op is already part of composite no need to do anything
  if (op->getParentOp() == composite)
    return nullptr;

  // If the op is not part of src, it will be materialized later
  if (op->getParentOp() != src.get())
    return nullptr;

  // TODO: The return value of linkGlobalValueProto should be an Expected type
  auto newProto = linkGlobalValueProto(op, false);
  if (!newProto) {
    // TODO: call setError
    return nullptr;
  }
  // TOOD: Extra check for the new value

  // TODO: Lots of if cases for Function, global variable, global alias.
  // for now, just check if it is a declaration, if so, not much more to do.
  if (auto gvl = dyn_cast<GlobalValueLinkageOpInterface>(newProto)) {
    if (!gvl.isDeclarationForLinkage()) {
      return newProto;
    }
  }

  // TODO: Some indirect symbol thing

  if (forIndirectSymbol || shouldLink(newProto, op))
    setError(linkGlobalValueBody(newProto, op));

  // TODO: Update attributes
  return newProto;
}

Error MLIRLinker::linkFunctionBody(Operation *dst,
                                   FunctionLinkageOpInterface src) {

  // TODO: Not exactly like this
  if (auto dstGVL = dyn_cast<GlobalValueLinkageOpInterface>(dst))
    assert(dstGVL.isDeclarationForLinkage());

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

Error MLIRLinker::linkGlobalValueBody(Operation *dst, Operation *src) {
  // TODO: Switch on what type of thing it is. For now just assume it is a
  // function-like thing
  if (auto f = dyn_cast<FunctionLinkageOpInterface>(src))
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
    auto op = worklist.back();
    worklist.pop_back();

    if (valueMap.contains(op)) // TODO: There is an indirect symbol value map,
                               // do we need that?
      continue;

    auto gvl = dyn_cast<GlobalValueLinkageOpInterface>(op);
    if (gvl)
      assert(!gvl.isDeclarationForLinkage());

    // TODO: Is this the equivalent of Mapper.mapValue?
    materialize(op);

    if (foundError)
      return std::move(*foundError);
    flushRAUWorklist();
  }

  // TODO: Do we need to reorder the globals?

  return Error::success();
}
} // namespace

IRMover::IRMover(Operation *composite) : composite(composite) {}

Error IRMover::move(OwningOpRef<Operation *> src,
                    ArrayRef<Operation *> valuesToLink) {

  MLIRLinker linker(composite, std::move(src), valuesToLink);
  Error e = linker.run();

  // TODO: Remove
  if (failed(verify(composite, true))) {
    llvm::outs() << "Verify failed\n";
  }
  return e;
}
