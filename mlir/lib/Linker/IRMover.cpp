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

#define DEBUG_TYPE "mlir-link-ir-mover"

using namespace mlir;
using namespace mlir::link;

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

  void scheduleRemapFunction(Function f) {
    worklist.push_back(f.getOperation());
  }

  Operation *mapSymbol(StringRef sym) {
    // TODO: This is special as we only have a symbol ref and need to get a
    // value currently implemented by asking the materializer. Might not be
    // exactly how it shall be done, but works for now.
    if (auto op = materializer.getSourceOperation(sym))
      return mapValue(op);

    return {};
  }

  Operation *mapValue(GlobalValue v) {
    Flusher f(*this);

    Operation *vo = v.getOperation();
    // If the value already exists in the map, use it.
    if (auto op = valueMap.lookupOrNull(vo)) {
      return op;
    }

    // If we have a materializer and it can materialize a value, use that.
    if (auto newv =
            materializer.materialize(v, /* forIndirectSymbol= */ false)) {
      valueMap.map(vo, newv);
      return newv;
    }

    // Global values do not need to be seeded into the VM if they
    // are using the identity mapping.
    if (flags & RF_NullMapMissingGlobalValues)
      return {};
    valueMap.map(vo, vo);
    return vo;
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
  RemapFlags flags = RF_None;

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
  IRMapping indirectSymbolValueMap;

  ValueMapper<MLIRLinker> mapper;

  DenseSet<GlobalValue> valuesToLink;
  std::vector<GlobalValue> worklist;
  // Replace-all-uses-with worklist
  std::vector<std::pair<Operation *, Operation *>> rauwWorklist;

  /// The Error encountered during materialization. We use an Optional here to
  /// avoid needing to manage an unconsumed success value.
  std::optional<Error> err = std::nullopt;

  // NOTE: This is the ValueMapper flush
  void flush();

  bool doneLinkingBodies{false};

  void maybeAdd(GlobalValue val) {
    if (valuesToLink.insert(val).second)
      worklist.push_back(val);
  }

  Error linkFunctionBody(Function dst, Function src);
  Error linkGlobalVariable(GlobalVariable dst, GlobalVariable src);
  Error linkAliasAliasee(GlobalAlias dst, GlobalAlias src);
  Error linkIFuncResolver(GlobalIFunc dst, GlobalIFunc src);
  Error linkGlobalValueBody(GlobalValue dst, GlobalValue src);
  Expected<Operation *> linkAppendingVarProto(GlobalVariable dst,
                                              GlobalVariable src);
  bool shouldLink(GlobalValue dst, GlobalValue src);
  GlobalVariable copyGlobalVariableProto(GlobalVariable src);
  Function copyFunctionProto(Function src);
  GlobalValue copyGlobalValueProto(GlobalValue src, bool forDefinition);
  Expected<Operation *> linkGlobalValueProto(GlobalValue sgv,
                                             bool forIndirectSymbol);

  /// Perform "replace all uses with" operations. These work items need to be
  /// performed as part of materialization, but we postpone them to happen after
  /// materialization is done. The materializer called by ValueMapper is not
  /// expected to delete constants, as ValueMapper is holding pointers to some
  /// of them, but constant destruction may be indirectly triggered by RAUW.
  /// Hence, the need to move this out of the materialization call chain.
  void flushRAUWWorklist();

  /// Given a global in the source module, return the global in the
  /// destination module that is being linked to, if any.
  GlobalValue getLinkedToGlobal(GlobalValue sgv) {

    // If the source has no name it can't link.  If it has local linkage,
    // there is no name match-up going on.
    auto symOp = dyn_cast<SymbolOpInterface>(sgv.getOperation());
    if (!symOp)
      return {};

    auto sgvName = symOp.getName();
    if (sgvName.empty() || sgv.hasLocalLinkage())
      return {};

    // Otherwise see if we have a match in the destination module's symtab.
    // TODO: Should the SymbolTable be a member instead?
    SymbolTable syms(composite);
    Operation *dstOp = syms.lookup(sgvName);
    if (!dstOp)
      return {};

    auto dgv = dyn_cast<GlobalValue>(dstOp);
    if (!dgv)
      return {};

    // If we found a global with the same name in the dest module, but it has
    // internal linkage, we are really not doing any linkage here.
    if (dgv.hasLocalLinkage())
      return {};

    // If we found an intrinsic declaration with mismatching prototypes, we
    // probably had a nameclash. Don't use that version.
    if (auto fdgv = dyn_cast<Function>(sgv)) {
      // TODO: Can we check for intrinsic functions?
    }

    // Otherwise, we do in fact link to the destination global.
    return dgv;
  }

  void insertUnique(GlobalValue gv, Operation *dst) {
    // LLVM does global value renaming automatically. This is a workaround to
    // ensure we only insert unique values.

    Operation *op = gv.getOperation();
    StringRef name = gv.getLinkedName();

    SymbolTable symbols(dst);
    if (symbols.lookup(name)) {
      [[maybe_unused]] auto renamed = symbols.renameToUnique(op, {});
      LLVM_DEBUG(llvm::dbgs()
                 << "Renaming global value: " << name << " to "
                 << (succeeded(renamed) ? renamed->str() : "failed") << "\n");
    }

    // TODO: make insertion point configurable (let mover keep insertion point)
    dst->getRegion(0).back().push_back(op);
  }

public:
  MLIRLinker(Operation *composite, OwningOpRef<Operation *> srcOp,
             ArrayRef<GlobalValue> valuesToLink)
      : composite(composite), src(std::move(srcOp)), mapper(valueMap, *this) {
    for (GlobalValue gvl : valuesToLink)
      maybeAdd(gvl);
  }

  // TODO: Helper function for the materializer to convert a symbol to a
  // linkable value
  GlobalValue getSourceOperation(StringRef sym) {
    SymbolTable syms(*src);
    if (auto op = syms.lookup(sym))
      return dyn_cast<GlobalValue>(op);
    return {};
  }

  Error run();

  Operation *materialize(GlobalValue gv, bool forIndirectSymbol);
};

/// TODO: check how this applies to MLIR
/// The LLVM SymbolTable class autorenames globals that conflict in the symbol
/// table. This is good for all clients except for us. Go through the trouble
/// to force this back.
static void forceRenaming(GlobalValue gv, StringRef name) {
  // If the global doesn't force its name or if it already has the right name,
  // there is nothing for us to do.
  if (gv.hasLocalLinkage() || gv.getLinkedName() == name)
    return;

  Operation *op = gv.getOperation();
  SymbolTable symbols(op->getParentOfType<LinkableModuleOpInterface>());

  // If there is a conflict, rename the conflict.
  if (Operation *conflict = symbols.lookup(name)) {
    llvm_unreachable("Not implemented");
  } else {
    gv.setLinkedName(name); // Force the name back
  }
}

bool MLIRLinker::shouldLink(GlobalValue dgv, GlobalValue sgv) {
  if (valuesToLink.count(sgv) || sgv.hasLocalLinkage()) {
    return true;
  }

  if (dgv && !dgv.isDeclarationForLinker())
    return false;

  if (sgv.isDeclaration() || doneLinkingBodies)
    return false;

  // Callback to the client to give a chance to lazily add the Global to the
  // list of value to link.
  bool lazilyAdded = false;
  //   if (AddLazyFor)
  //     AddLazyFor(SGV, [this, &LazilyAdded](GlobalValue &GV) {
  //       maybeAdd(&GV);
  //       LazilyAdded = true;
  //     });
  // TODO: Implement callback, if needed
  return lazilyAdded;
}

// TODO: move to linker interface
GlobalVariable MLIRLinker::copyGlobalVariableProto(GlobalVariable sgv) {
  // No linking to be performed or linking from the source: simply create an
  // identical version of the symbol over in the dest module... the
  // initializer will be filled in later by LinkGlobalInits.
  GlobalVariable gv = cast<GlobalVariable>(src->cloneWithoutRegions());
  gv.setLinkage(link::Linkage::External);
  gv.setAlignment(sgv.getAlignment());
  insertUnique(gv, composite);
  return gv;
}

Function MLIRLinker::copyFunctionProto(Function src) {
  // Clone the operation (without regions to ensure it becomes empty as is
  // considered a decl)

  // TODO: make this part of linker interface
  Function func = cast<FunctionLinkageOpInterface>(
      src.getOperation()->cloneWithoutRegions());
  insertUnique(func, composite);
  return func;
}

GlobalValue MLIRLinker::copyGlobalValueProto(GlobalValue sgv,
                                             bool forDefinition) {
  GlobalValue newgv;
  if (auto sgvar = dyn_cast<GlobalVariable>(sgv)) {
    newgv = copyGlobalVariableProto(sgvar);
  } else if (auto sf = dyn_cast<Function>(sgv)) {
    newgv = copyFunctionProto(sf);
  } else {
    llvm_unreachable("Not implemented");
  }

  if (forDefinition) {
    newgv.setLinkage(sgv.getLinkage());
  } else if (sgv.hasExternalWeakLinkage()) {
    newgv.setLinkage(link::Linkage::ExternWeak);
  }

  // TODO copy metadata?

  return newgv;
}

Expected<Operation *> MLIRLinker::linkGlobalValueProto(GlobalValue sgv,
                                                       bool forIndirectSymbol) {
  GlobalValue dgv = getLinkedToGlobal(sgv);

  bool shouldLinkOps = shouldLink(dgv, sgv);

  // just missing from map
  if (shouldLinkOps) {
    Operation *sgvop = sgv.getOperation();
    if (Operation *inDst = valueMap.lookupOrNull(sgvop))
      return inDst;
    if (Operation *inDst = indirectSymbolValueMap.lookupOrNull(sgvop))
      return inDst;
  }

  if (!shouldLinkOps && forIndirectSymbol)
    dgv = {};

  // Handle the ultra special appending linkage case first.
  if (sgv.hasAppendingLinkage() || (dgv && dgv.hasAppendingLinkage())) {
    auto sgvar = cast<GlobalVariable>(sgv);
    auto dgvar = cast_or_null<GlobalVariable>(dgv);
    return linkAppendingVarProto(dgvar, sgvar);
  }

  bool needsRenaming = false;
  GlobalValue newgv;
  if (dgv && !shouldLinkOps) {
    newgv = dgv;
  } else {
    // If we are done linking global value bodies (i.e. we are performing
    // metadata linking), don't link in the global value due to this
    // reference, simply map it to null.
    if (doneLinkingBodies)
      return nullptr;

    newgv = copyGlobalValueProto(sgv, shouldLinkOps || forIndirectSymbol);
    if (shouldLinkOps || !forIndirectSymbol)
      needsRenaming = true;
  }

  if (isa<Function>(newgv.getOperation())) {
    // TODO: overloaded intrinsics
  }

  if (needsRenaming) {
    forceRenaming(newgv, sgv.getLinkedName());
  }

  if (shouldLinkOps || forIndirectSymbol) {
    if (auto sc = sgv.getComdatPair()) {
      llvm_unreachable("Not implemented");
    }
  }

  if (!shouldLinkOps && forIndirectSymbol)
    newgv.setLinkage(link::Linkage::Internal);

  // TODO: bitcasts needed??

  if (dgv && newgv.getOperation() != dgv.getOperation()) {
    // Schedule "replace all uses with" to happen after materializing is
    // done. It is not safe to do it now, since ValueMapper may be holding
    // pointers to constants that will get deleted if RAUW runs.
    rauwWorklist.push_back(
        std::make_pair(dgv.getOperation(), newgv.getOperation()));
  }

  return newgv.getOperation();
}

Operation *MLIRLinker::materialize(GlobalValue sgv, bool forIndirectSymbol) {
  LLVM_DEBUG(llvm::dbgs() << "Materializing: " << sgv.getLinkedName() << "\n");
  // If v is from dst, it was already materialized when dst was loaded.
  if (composite->isProperAncestor(sgv.getOperation())) {
    LLVM_DEBUG(llvm::dbgs() << "  already materialized in dst\n");
    return {};
  }

  // When linking a global from other modules than src and dst, skip
  // materializing it because it would be mapped later when its containing
  // module is linked. Linking it now would potentially pull in many types
  // that may not be mapped properly.
  if (!src->isProperAncestor(sgv.getOperation())) {
    LLVM_DEBUG(llvm::dbgs() << "  skipping materialization from non-src\n");
    return {};
  }

  auto newProto = linkGlobalValueProto(sgv, forIndirectSymbol);
  if (!newProto) {
    err = newProto.takeError();
    return {};
  }

  if (!*newProto)
    return {};

  GlobalValue newgv = dyn_cast<GlobalValue>(*newProto);
  if (!newgv)
    return *newProto;

  auto newop = newgv.getOperation();
  // If we already created the body, just return.
  // TODO this should be just `hasMaterializedBody` in linker interface
  // This is too llvm specific
  if (auto f = dyn_cast<Function>(newgv)) {
    if (!f.isDeclaration())
      return newop;
  } else if (auto var = dyn_cast<GlobalVariable>(newgv)) {
    if (!var.isDeclaration() || var.hasAppendingLinkage())
      return newop;
  } else if (auto ga = dyn_cast<GlobalAlias>(newgv)) {
    if (ga.getAliasee())
      return newop;
  } else if (auto gi = dyn_cast<GlobalIFunc>(newgv)) {
    if (gi.getResolver())
      return newop;
  } else {
    llvm_unreachable("Not implemented");
  }

  // If the global is being linked for an indirect symbol, it may have already
  // been scheduled to satisfy a regular symbol. Similarly, a global being
  // linked for a regular symbol may have already been scheduled for an indirect
  // symbol. Check for these cases by looking in the other value map and
  // confirming the same value has been scheduled.  If there is an entry in the
  // ValueMap but the value is different, it means that the value already had a
  // definition in the destination module (linkonce for instance), but we need a
  // new definition for the indirect symbol ("New" will be different).
  Operation *sgvop = sgv.getOperation();
  if ((forIndirectSymbol && valueMap.lookupOrNull(sgvop)) ||
      (!forIndirectSymbol &&
       indirectSymbolValueMap.lookupOrNull(sgvop) == newop)) {
    return newop;
  }

  if (forIndirectSymbol || shouldLink(newgv, sgv))
    err = linkGlobalValueBody(newgv, sgv);
  return newop;
}

/// Update the initializers in the Dest module now that all globals that may
/// be referenced are in Dest.
Error MLIRLinker::linkGlobalVariable(GlobalVariable dst, GlobalVariable src) {
  // Figure out what the initializer looks like in the dest module.
  // TODO: Schedule global init
  // TODO: This will likely only need to happen for those that have an
  // initializer, not for constants
  llvm_unreachable("Not implemented");
}

Error MLIRLinker::linkAliasAliasee(GlobalAlias dst, GlobalAlias src) {
  llvm_unreachable("Not implemented");
}

Error MLIRLinker::linkIFuncResolver(GlobalIFunc dst, GlobalIFunc src) {
  llvm_unreachable("Not implemented");
}

/// Copy the source function over into the dest function and fix up references
/// to values. At this point we know that Dest is an external function, and
/// that Src is not.
Error MLIRLinker::linkFunctionBody(Function dst, Function src) {
  assert(dst.isDeclaration() && !src.isDeclaration());

  // TODO materialize src if needed

  // TODO deal with prefix data?
  // TODO deal with prologue data?
  // TODO deal with personality function?

  Operation *srcOp = src.getOperation();
  Operation *dstOp = dst.getOperation();
  assert(srcOp->getNumRegions() == dstOp->getNumRegions() &&
         "Operations must have same number of regions");

  for (auto [srcRegion, dstRegion] :
       llvm::zip(srcOp->getRegions(), dstOp->getRegions())) {
    dstRegion.takeBody(srcRegion);
  }

  mapper.scheduleRemapFunction(dst);
  return Error::success();
}

// TODO: this should be abstracted as linkGlobalBody interface method
Error MLIRLinker::linkGlobalValueBody(GlobalValue dst, GlobalValue src) {
  if (auto f = dyn_cast<Function>(src))
    return linkFunctionBody(cast<Function>(dst), f);
  if (auto gv = dyn_cast<GlobalVariable>(src.getOperation()))
    return linkGlobalVariable(cast<GlobalVariable>(dst), gv);
  if (auto ga = dyn_cast<GlobalAlias>(src.getOperation()))
    return linkAliasAliasee(cast<GlobalAlias>(dst), ga);
  return linkIFuncResolver(cast<GlobalIFunc>(dst), cast<GlobalIFunc>(src));
}

Expected<Operation *> MLIRLinker::linkAppendingVarProto(GlobalVariable dst,
                                                        GlobalVariable src) {
  llvm_unreachable("Not implemented");
}

void MLIRLinker::flushRAUWWorklist() {
  for (const auto &[oldOp, newOp] : rauwWorklist) {
    if (auto sym = dyn_cast<SymbolOpInterface>(newOp)) {
      auto name = sym.getNameAttr();
      if (failed(SymbolTable::replaceAllSymbolUses(oldOp, name, composite)))
        oldOp->emitError("unable to replace all symbol uses for ") << name;
    }
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

  std::reverse(worklist.begin(), worklist.end());
  while (!worklist.empty()) {
    GlobalValue gv = worklist.back();
    worklist.pop_back();

    LLVM_DEBUG(llvm::dbgs() << "Linking: " << gv.getLinkedName() << "\n");

    Operation *gvo = gv.getOperation();
    if (valueMap.contains(gvo) || indirectSymbolValueMap.contains(gvo))
      continue;

    assert(!gv.isDeclaration());

    mapper.mapValue(gv);

    if (err)
      return std::move(*err);
    flushRAUWWorklist();
  }

  doneLinkingBodies = true;
  mapper.addFlags(RF_NullMapMissingGlobalValues);

  // Reorder the globals just added to the destination module to match their
  // original order in the source module.
  src->walk([&](GlobalVariable gv) {
    if (gv.hasAppendingLinkage())
      return WalkResult::skip();
    if (auto newValue = mapper.mapValue(gv)) {
      if (auto newGv = dyn_cast<GlobalVariable>(newValue)) {
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
                    ArrayRef<GlobalValue> valuesToLink) {

  MLIRLinker linker(composite, std::move(src), valuesToLink);
  return linker.run();
}
