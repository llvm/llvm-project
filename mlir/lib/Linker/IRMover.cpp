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

#include <assert.h>
#include <optional>

using namespace mlir;
using llvm::Error;
using llvm::Expected;

namespace {

// TODO: Should this exist?
enum RemapFlags {
  RF_None = 0,
  RF_NullMapMissingGlobalValues = 8,
};

inline RemapFlags operator|(RemapFlags LHS, RemapFlags RHS) {
  return RemapFlags(unsigned(LHS) | unsigned(RHS));
}

// NOTE: This is a simplified version of the LLVM IR one.
template <typename T>
class ValueMapper {
public:
  // TODO: Consider creating an interface for the materializer
  ValueMapper(IRMapping &valueMap, T &materializer)
      : valueMap(valueMap), materializer(materializer) {}

  // TODO: Maybe this should be called remapValue?
  void scheduleRemapFunction(Operation *v) { worklist.push_back(v); }

  Operation *mapSymbol(StringRef sym) {
    // TODO: This is special as we only have a symbol ref and need to get a
    // value currently implemented by asking the materializer. Might not be
    // exactly how it shall be done, but works for now.
    if (auto op = materializer.getSourceOperation(sym))
      return mapValue(op);

    return nullptr;
  }

  Operation *mapValue(Operation *v) {
    Flusher f(*this);

    // If the value already exists in the map, use it.
    if (auto op = valueMap.lookupOrNull(v)) {
      return op;
    }

    // If we have a materializer and it can materialize a value, use that.
    if (auto newv = materializer.materialize(v, false)) {
      valueMap.map(v, newv);
      return newv;
    }

    // Global values do not need to be seeded into the VM if they
    // are using the identity mapping.
    if (auto gvl = dyn_cast<GlobalValueLinkageOpInterface>(v)) {
      if (flags & RF_NullMapMissingGlobalValues)
        return nullptr;
      valueMap.map(v, v);
      return v;
    }
    // TODO: potentially only value mapping

    // TODO: Inline asm

    // TODO: MetadataAsValue??

    // TODO: Constants etc.
    assert(false);
  };

  bool hasWorkToDo() const { return !worklist.empty(); }

  void addFlags(RemapFlags additionalFlags) {
    assert(!hasWorkToDo() && "Expected to have flushed the worklist");
    flags = flags | additionalFlags;
  }

  void remapFunction(Operation *f) {
    // TODO: Remap operands

    f->walk([&](Operation *op) { remapInstruction(op); });

    // Remap the metadata attachments.
    // TODO: Do we need to do this?
  }

  void remapInstruction(Operation *op) {
    Flusher f(*this);

    // TODO: Operands?

    if (auto symuser = dyn_cast<SymbolUserOpInterface>(op)) {
      if (auto sym = symuser.getUserSymbol()) {
        mapSymbol(*sym);
      }
    }
  }

  void flush() {
    // Flush out the worklist of global values.
    while (!worklist.empty()) {
      auto e = worklist.pop_back_val();

      // TODO: for now, we only handle functions
      remapFunction(e);
    }
  }

private:
  struct Flusher {
    ValueMapper<T> &vm;
    Flusher(ValueMapper<T> &vm) : vm(vm) {}
    ~Flusher() { vm.flush(); }
  };

  IRMapping &valueMap;
  T &materializer;
  RemapFlags flags;

  // TODO: The worklist is just a function linkage opfor now.
  // Once we add global value init we will need to extend it.
  SmallVector<Operation *, 4> worklist;
};

class MLIRLinker {
  Operation *composite;
  OwningOpRef<Operation *> src;

  // TODO: This is a ValueToValueMapTy in llvm-link. I'm assuming this would be
  // the equivalent in mlir.
  IRMapping valueMap;

  ValueMapper<MLIRLinker> mapper;

  DenseSet<GlobalValueLinkageOpInterface> valuesToLink;
  std::vector<GlobalValueLinkageOpInterface> worklist;
  // Replace-all-uses-with worklist
  std::vector<std::pair<Operation *, Operation *>> rauwWorklist;

  // NOTE: This is the ValueMapper flush
  void flush();

  bool doneLinkingBodies{false};

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
  void linkGlobalVariable(Operation *dst, GlobalVariableLinkageOpInterface src);
  Error linkGlobalValueBody(Operation *dst, GlobalValueLinkageOpInterface src);
  bool shouldLink(Operation *dst, Operation *src);
  Operation *copyGlobalVariableProto(Operation *src);
  Operation *copyFunctionProto(Operation *src);
  Operation *copyGlobalValueProto(Operation *src, bool forDefinition);
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

  void insertUnique(Operation *op, Operation *dst) {
    // LLVM does global value renaming automatically. This is a workaround to
    // ensure we only insert unique values.
    bool needsRename = false;
    if (auto gv = dyn_cast<SymbolOpInterface>(op)) {
      auto name = gv.getName();
      SymbolTable syms(dst);
      if (syms.lookup(name)) {
        (void)syms.renameToUnique(op, {});
      }
    }

    OpBuilder b(dst->getRegion(0));
    b.insert(op);
  }

public:
  MLIRLinker(Operation *composite, OwningOpRef<Operation *> srcOp,
             ArrayRef<GlobalValueLinkageOpInterface> valuesToLink)
      : composite{composite}, src{std::move(srcOp)}, mapper{valueMap, *this} {
    for (GlobalValueLinkageOpInterface gvl : valuesToLink)
      maybeAdd(gvl);
  }

  // TODO: Helper function for the materializer to convert a symbol to a
  // linkable value
  GlobalValueLinkageOpInterface getSourceOperation(StringRef sym) {
    SymbolTable syms(*src);
    if (auto op = syms.lookup(sym))
      return dyn_cast<GlobalValueLinkageOpInterface>(op);
    return nullptr;
  }

  Error run();

  Operation *materialize(Operation *v, bool forIndirectSymbol);
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

  if (sgv.isDeclarationForLinkage() || doneLinkingBodies)
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

Operation *MLIRLinker::copyGlobalVariableProto(Operation *src) {
  // No linking to be performed or linking from the source: simply create an
  // identical version of the symbol over in the dest module... the
  // initializer will be filled in later by LinkGlobalInits.
  // OpBuilder builder(composite->getRegion(0));
  auto sgv = dyn_cast<GlobalVariableLinkageOpInterface>(src);
  Operation *newGv = src->cloneWithoutRegions();
  if (auto gv = dyn_cast<GlobalVariableLinkageOpInterface>(newGv)) {
    gv.setLinkage(link::Linkage::External);
    gv.setAlignment(sgv.getAlignment());
  }
  insertUnique(newGv, composite);
  return newGv;
}

Operation *MLIRLinker::copyFunctionProto(Operation *src) {
  // Clone the operation (without regions to ensure it becomes empty as is
  // considered a decl)
  Operation *newFunc = src->cloneWithoutRegions();
  insertUnique(newFunc, composite);
  return newFunc;
}
Operation *MLIRLinker::copyGlobalValueProto(Operation *src,
                                            bool forDefinition) {
  // TODO: Change signature to accept GlobalValueLinkageOpInterface
  auto sgv = dyn_cast<GlobalValueLinkageOpInterface>(src);
  Operation *newOp;
  if (auto sgvar = dyn_cast<GlobalVariableLinkageOpInterface>(src)) {
    newOp = copyGlobalVariableProto(src);
  } else if (auto f = dyn_cast<FunctionLinkageOpInterface>(src)) {
    newOp = copyFunctionProto(src);
  }
  auto newGv = dyn_cast<GlobalValueLinkageOpInterface>(newOp);
  // TODO: This is unfortunate. Conside changing the return types.
  if (!newGv)
    return newGv;

  if (forDefinition || sgv.hasExternalWeakLinkage())
    newGv.setLinkage(sgv.getLinkage());

  // TODO: Copy metadata for global variables and function declarations?

  // TODO: If function clear personality, prefix and prologue data
  return newGv;
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
    // If we are done linking global value bodies (i.e. we are performing
    // metadata linking), don't link in the global value due to this
    // reference, simply map it to null.
    if (doneLinkingBodies)
      return nullptr;

    newDst = copyGlobalValueProto(sgv.getOperation(),
                                  shouldLinkOps || forIndirectSymbol);
    if (shouldLinkOps) // TODO: || !ForIndirectSymbol?
      needsRenaming = true;
  }

  // TODO: overloaded intrinsics

  // if (needsRenaming)
  //   forceRenaming(newDst, srcName);

  // TODO: Comdat

  if (!shouldLinkOps && forIndirectSymbol)
    if (auto newDstGVL = dyn_cast<GlobalValueLinkageOpInterface>(newDst))
      newDstGVL.setLinkage(link::Linkage::Internal);

  // TODO: bitcasts needed??

  if (dgv && newDst != dgv.getOperation()) {
    // Schedule "replace all uses with"
    rauwWorklist.push_back(
        std::make_pair(dgv.getOperation(), newDst)); // TODO: This is simplified
  }

  return newDst;
}

Operation *MLIRLinker::materialize(Operation *v, bool forIndirectSymbol) {
  auto sgv = dyn_cast<GlobalValueLinkageOpInterface>(v);
  if (!sgv)
    return nullptr;

  // If v is from dest, it was already materialized when dest was loaded.
  if (v->getParentOp() == composite)
    return nullptr;

  // When linking a global from other modules than source & dest, skip
  // materializing it because it would be mapped later when its containing
  // module is linked. Linking it now would potentially pull in many types that
  // may not be mapped properly.
  if (v->getParentOp() != src.get())
    return nullptr;

  auto newProto = linkGlobalValueProto(sgv, false);
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
  } else if (auto var = dyn_cast<GlobalVariableLinkageOpInterface>(
                 newGvl.getOperation())) {
    if (!var.isDeclarationForLinkage() || var.hasAppendingLinkage())
      return *newProto;
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

  if (forIndirectSymbol || shouldLink(newGvl.getOperation(), v))
    setError(linkGlobalValueBody(newGvl.getOperation(), sgv));

  // TODO: Update attributes
  return newGvl.getOperation();
}

/// Update the initializers in the Dest module now that all globals that may be
/// referenced are in Dest.
void MLIRLinker::linkGlobalVariable(Operation *dst,
                                    GlobalVariableLinkageOpInterface src) {
  // Figure out what the initializer looks like in the dest module.
  // TODO: Schedule global init
  // TODO: This will likely only need to happen for those that have an
  // initializer, not for constants
}

/// Copy the source function over into the dest function and fix up references
/// to values. At this point we know that Dest is an external function, and
/// that Src is not.
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
  mapper.scheduleRemapFunction(dst);

  // auto target = src.getOperation()->clone(mapping);
  // dst->replaceAllUsesWith(target);
  // dst is an external sym and src is not
  return Error::success();
}

Error MLIRLinker::linkGlobalValueBody(Operation *dst,
                                      GlobalValueLinkageOpInterface src) {
  if (auto f = dyn_cast<FunctionLinkageOpInterface>(src.getOperation()))
    return linkFunctionBody(dst, f);
  if (auto gvar =
          dyn_cast<GlobalVariableLinkageOpInterface>(src.getOperation())) {
    linkGlobalVariable(dst, gvar);
    return Error::success();
  }

  return Error::success();
}

void MLIRLinker::flushRAUWorklist() {
  SymbolTable syms(composite);
  for (const auto &elem : rauwWorklist) {
    Operation *oldOp, *newOp;
    std::tie(oldOp, newOp) = elem;
    if (auto sym = dyn_cast<SymbolOpInterface>(newOp))
      syms.replaceAllSymbolUses(oldOp, sym.getNameAttr(), composite);
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

    mapper.mapValue(gvl);

    if (foundError)
      return std::move(*foundError);
    flushRAUWorklist();
  }

  doneLinkingBodies = true;
  mapper.addFlags(RF_NullMapMissingGlobalValues);

  // Reorder the globals just added to the destination module to match their
  // original order in the source module.
  src->walk([&](GlobalVariableLinkageOpInterface gv) {
    if (gv.hasAppendingLinkage())
      return WalkResult::skip();
    if (auto newValue = mapper.mapValue(gv)) {
      if (auto newGv = dyn_cast<GlobalVariableLinkageOpInterface>(newValue)) {
        newGv->remove();
        composite->getRegion(0).back().push_back(newGv);
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

  return e;
}
