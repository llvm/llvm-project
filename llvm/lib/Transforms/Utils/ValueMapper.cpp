//===- ValueMapper.cpp - Interface shared by lib/Transforms/Utils ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MapValue function, which is shared by various parts of
// the lib/Transforms/Utils library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Operator.h"
using namespace llvm;

// Out of line method to get vtable etc for class.
void ValueMapTypeRemapper::anchor() {}
void ValueMaterializer::anchor() {}
void ValueMaterializer::materializeInitFor(GlobalValue *New, GlobalValue *Old) {
}

Value *llvm::MapValue(const Value *V, ValueToValueMapTy &VM, RemapFlags Flags,
                      ValueMapTypeRemapper *TypeMapper,
                      ValueMaterializer *Materializer) {
  ValueToValueMapTy::iterator I = VM.find(V);
  
  // If the value already exists in the map, use it.
  if (I != VM.end() && I->second) return I->second;
  
  // If we have a materializer and it can materialize a value, use that.
  if (Materializer) {
    if (Value *NewV =
            Materializer->materializeDeclFor(const_cast<Value *>(V))) {
      VM[V] = NewV;
      if (auto *NewGV = dyn_cast<GlobalValue>(NewV))
        Materializer->materializeInitFor(
            NewGV, const_cast<GlobalValue *>(cast<GlobalValue>(V)));
      return NewV;
    }
  }

  // Global values do not need to be seeded into the VM if they
  // are using the identity mapping.
  if (isa<GlobalValue>(V)) {
    if (Flags & RF_NullMapMissingGlobalValues) {
      assert(!(Flags & RF_IgnoreMissingEntries) &&
             "Illegal to specify both RF_NullMapMissingGlobalValues and "
             "RF_IgnoreMissingEntries");
      return nullptr;
    }
    return VM[V] = const_cast<Value*>(V);
  }

  if (const InlineAsm *IA = dyn_cast<InlineAsm>(V)) {
    // Inline asm may need *type* remapping.
    FunctionType *NewTy = IA->getFunctionType();
    if (TypeMapper) {
      NewTy = cast<FunctionType>(TypeMapper->remapType(NewTy));

      if (NewTy != IA->getFunctionType())
        V = InlineAsm::get(NewTy, IA->getAsmString(), IA->getConstraintString(),
                           IA->hasSideEffects(), IA->isAlignStack());
    }
    
    return VM[V] = const_cast<Value*>(V);
  }

  if (const auto *MDV = dyn_cast<MetadataAsValue>(V)) {
    const Metadata *MD = MDV->getMetadata();
    // If this is a module-level metadata and we know that nothing at the module
    // level is changing, then use an identity mapping.
    if (!isa<LocalAsMetadata>(MD) && (Flags & RF_NoModuleLevelChanges))
      return VM[V] = const_cast<Value *>(V);

    auto *MappedMD = MapMetadata(MD, VM, Flags, TypeMapper, Materializer);
    if (MD == MappedMD || (!MappedMD && (Flags & RF_IgnoreMissingEntries)))
      return VM[V] = const_cast<Value *>(V);

    // FIXME: This assert crashes during bootstrap, but I think it should be
    // correct.  For now, just match behaviour from before the metadata/value
    // split.
    //
    //    assert((MappedMD || (Flags & RF_NullMapMissingGlobalValues)) &&
    //           "Referenced metadata value not in value map");
    return VM[V] = MetadataAsValue::get(V->getContext(), MappedMD);
  }

  // Okay, this either must be a constant (which may or may not be mappable) or
  // is something that is not in the mapping table.
  Constant *C = const_cast<Constant*>(dyn_cast<Constant>(V));
  if (!C)
    return nullptr;
  
  if (BlockAddress *BA = dyn_cast<BlockAddress>(C)) {
    Function *F = 
      cast<Function>(MapValue(BA->getFunction(), VM, Flags, TypeMapper, Materializer));
    BasicBlock *BB = cast_or_null<BasicBlock>(MapValue(BA->getBasicBlock(), VM,
                                                       Flags, TypeMapper, Materializer));
    return VM[V] = BlockAddress::get(F, BB ? BB : BA->getBasicBlock());
  }
  
  // Otherwise, we have some other constant to remap.  Start by checking to see
  // if all operands have an identity remapping.
  unsigned OpNo = 0, NumOperands = C->getNumOperands();
  Value *Mapped = nullptr;
  for (; OpNo != NumOperands; ++OpNo) {
    Value *Op = C->getOperand(OpNo);
    Mapped = MapValue(Op, VM, Flags, TypeMapper, Materializer);
    if (Mapped != C) break;
  }
  
  // See if the type mapper wants to remap the type as well.
  Type *NewTy = C->getType();
  if (TypeMapper)
    NewTy = TypeMapper->remapType(NewTy);

  // If the result type and all operands match up, then just insert an identity
  // mapping.
  if (OpNo == NumOperands && NewTy == C->getType())
    return VM[V] = C;
  
  // Okay, we need to create a new constant.  We've already processed some or
  // all of the operands, set them all up now.
  SmallVector<Constant*, 8> Ops;
  Ops.reserve(NumOperands);
  for (unsigned j = 0; j != OpNo; ++j)
    Ops.push_back(cast<Constant>(C->getOperand(j)));
  
  // If one of the operands mismatch, push it and the other mapped operands.
  if (OpNo != NumOperands) {
    Ops.push_back(cast<Constant>(Mapped));
  
    // Map the rest of the operands that aren't processed yet.
    for (++OpNo; OpNo != NumOperands; ++OpNo)
      Ops.push_back(MapValue(cast<Constant>(C->getOperand(OpNo)), VM,
                             Flags, TypeMapper, Materializer));
  }
  Type *NewSrcTy = nullptr;
  if (TypeMapper)
    if (auto *GEPO = dyn_cast<GEPOperator>(C))
      NewSrcTy = TypeMapper->remapType(GEPO->getSourceElementType());

  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C))
    return VM[V] = CE->getWithOperands(Ops, NewTy, false, NewSrcTy);
  if (isa<ConstantArray>(C))
    return VM[V] = ConstantArray::get(cast<ArrayType>(NewTy), Ops);
  if (isa<ConstantStruct>(C))
    return VM[V] = ConstantStruct::get(cast<StructType>(NewTy), Ops);
  if (isa<ConstantVector>(C))
    return VM[V] = ConstantVector::get(Ops);
  // If this is a no-operand constant, it must be because the type was remapped.
  if (isa<UndefValue>(C))
    return VM[V] = UndefValue::get(NewTy);
  if (isa<ConstantAggregateZero>(C))
    return VM[V] = ConstantAggregateZero::get(NewTy);
  assert(isa<ConstantPointerNull>(C));
  return VM[V] = ConstantPointerNull::get(cast<PointerType>(NewTy));
}

static Metadata *mapToMetadata(ValueToValueMapTy &VM, const Metadata *Key,
                               Metadata *Val, ValueMaterializer *Materializer,
                               RemapFlags Flags) {
  VM.MD()[Key].reset(Val);
  if (Materializer && !(Flags & RF_HaveUnmaterializedMetadata)) {
    auto *N = dyn_cast_or_null<MDNode>(Val);
    // Need to invoke this once we have non-temporary MD.
    if (!N || !N->isTemporary())
      Materializer->replaceTemporaryMetadata(Key, Val);
  }
  return Val;
}

static Metadata *mapToSelf(ValueToValueMapTy &VM, const Metadata *MD,
                           ValueMaterializer *Materializer, RemapFlags Flags) {
  return mapToMetadata(VM, MD, const_cast<Metadata *>(MD), Materializer, Flags);
}

static Metadata *MapMetadataImpl(const Metadata *MD,
                                 SmallVectorImpl<MDNode *> &DistinctWorklist,
                                 ValueToValueMapTy &VM, RemapFlags Flags,
                                 ValueMapTypeRemapper *TypeMapper,
                                 ValueMaterializer *Materializer);

static Metadata *mapMetadataOp(Metadata *Op,
                               SmallVectorImpl<MDNode *> &DistinctWorklist,
                               ValueToValueMapTy &VM, RemapFlags Flags,
                               ValueMapTypeRemapper *TypeMapper,
                               ValueMaterializer *Materializer) {
  if (!Op)
    return nullptr;

  if (Materializer && !Materializer->isMetadataNeeded(Op))
    return nullptr;

  if (Metadata *MappedOp = MapMetadataImpl(Op, DistinctWorklist, VM, Flags,
                                           TypeMapper, Materializer))
    return MappedOp;
  // Use identity map if MappedOp is null and we can ignore missing entries.
  if (Flags & RF_IgnoreMissingEntries)
    return Op;

  // FIXME: This assert crashes during bootstrap, but I think it should be
  // correct.  For now, just match behaviour from before the metadata/value
  // split.
  //
  //    assert((Flags & RF_NullMapMissingGlobalValues) &&
  //           "Referenced metadata not in value map!");
  return nullptr;
}

/// Resolve uniquing cycles involving the given metadata.
static void resolveCycles(Metadata *MD, bool AllowTemps) {
  if (auto *N = dyn_cast_or_null<MDNode>(MD)) {
    if (AllowTemps && N->isTemporary())
      return;
    if (!N->isResolved())
      N->resolveCycles(AllowTemps);
  }
}

/// Remap the operands of an MDNode.
///
/// If \c Node is temporary, uniquing cycles are ignored.  If \c Node is
/// distinct, uniquing cycles are resolved as they're found.
///
/// \pre \c Node.isDistinct() or \c Node.isTemporary().
static bool remapOperands(MDNode &Node,
                          SmallVectorImpl<MDNode *> &DistinctWorklist,
                          ValueToValueMapTy &VM, RemapFlags Flags,
                          ValueMapTypeRemapper *TypeMapper,
                          ValueMaterializer *Materializer) {
  assert(!Node.isUniqued() && "Expected temporary or distinct node");
  const bool IsDistinct = Node.isDistinct();

  bool AnyChanged = false;
  for (unsigned I = 0, E = Node.getNumOperands(); I != E; ++I) {
    Metadata *Old = Node.getOperand(I);
    Metadata *New = mapMetadataOp(Old, DistinctWorklist, VM, Flags, TypeMapper,
                                  Materializer);
    if (Old != New) {
      AnyChanged = true;
      Node.replaceOperandWith(I, New);

      // Resolve uniquing cycles underneath distinct nodes on the fly so they
      // don't infect later operands.
      if (IsDistinct)
        resolveCycles(New, Flags & RF_HaveUnmaterializedMetadata);
    }
  }

  return AnyChanged;
}

/// Map a distinct MDNode.
///
/// Whether distinct nodes change is independent of their operands.  If \a
/// RF_MoveDistinctMDs, then they are reused, and their operands remapped in
/// place; effectively, they're moved from one graph to another.  Otherwise,
/// they're cloned/duplicated, and the new copy's operands are remapped.
static Metadata *mapDistinctNode(const MDNode *Node,
                                 SmallVectorImpl<MDNode *> &DistinctWorklist,
                                 ValueToValueMapTy &VM, RemapFlags Flags,
                                 ValueMapTypeRemapper *TypeMapper,
                                 ValueMaterializer *Materializer) {
  assert(Node->isDistinct() && "Expected distinct node");

  MDNode *NewMD;
  if (Flags & RF_MoveDistinctMDs)
    NewMD = const_cast<MDNode *>(Node);
  else
    NewMD = MDNode::replaceWithDistinct(Node->clone());

  // Remap operands later.
  DistinctWorklist.push_back(NewMD);
  return mapToMetadata(VM, Node, NewMD, Materializer, Flags);
}

/// \brief Map a uniqued MDNode.
///
/// Uniqued nodes may not need to be recreated (they may map to themselves).
static Metadata *mapUniquedNode(const MDNode *Node,
                                SmallVectorImpl<MDNode *> &DistinctWorklist,
                                ValueToValueMapTy &VM, RemapFlags Flags,
                                ValueMapTypeRemapper *TypeMapper,
                                ValueMaterializer *Materializer) {
  assert(((Flags & RF_HaveUnmaterializedMetadata) || Node->isUniqued()) &&
         "Expected uniqued node");

  // Create a temporary node and map it upfront in case we have a uniquing
  // cycle.  If necessary, this mapping will get updated by RAUW logic before
  // returning.
  auto ClonedMD = Node->clone();
  mapToMetadata(VM, Node, ClonedMD.get(), Materializer, Flags);
  if (!remapOperands(*ClonedMD, DistinctWorklist, VM, Flags, TypeMapper,
                     Materializer)) {
    // No operands changed, so use the original.
    ClonedMD->replaceAllUsesWith(const_cast<MDNode *>(Node));
    // Even though replaceAllUsesWith would have replaced the value map
    // entry, we need to explictly map with the final non-temporary node
    // to replace any temporary metadata via the callback.
    return mapToSelf(VM, Node, Materializer, Flags);
  }

  // Uniquify the cloned node. Explicitly map it with the final non-temporary
  // node so that replacement of temporary metadata via the callback occurs.
  return mapToMetadata(VM, Node,
                       MDNode::replaceWithUniqued(std::move(ClonedMD)),
                       Materializer, Flags);
}

static Metadata *MapMetadataImpl(const Metadata *MD,
                                 SmallVectorImpl<MDNode *> &DistinctWorklist,
                                 ValueToValueMapTy &VM, RemapFlags Flags,
                                 ValueMapTypeRemapper *TypeMapper,
                                 ValueMaterializer *Materializer) {
  // If the value already exists in the map, use it.
  if (Metadata *NewMD = VM.MD().lookup(MD).get())
    return NewMD;

  if (isa<MDString>(MD))
    return mapToSelf(VM, MD, Materializer, Flags);

  if (isa<ConstantAsMetadata>(MD))
    if ((Flags & RF_NoModuleLevelChanges))
      return mapToSelf(VM, MD, Materializer, Flags);

  if (const auto *VMD = dyn_cast<ValueAsMetadata>(MD)) {
    Value *MappedV =
        MapValue(VMD->getValue(), VM, Flags, TypeMapper, Materializer);
    if (VMD->getValue() == MappedV ||
        (!MappedV && (Flags & RF_IgnoreMissingEntries)))
      return mapToSelf(VM, MD, Materializer, Flags);

    // FIXME: This assert crashes during bootstrap, but I think it should be
    // correct.  For now, just match behaviour from before the metadata/value
    // split.
    //
    //    assert((MappedV || (Flags & RF_NullMapMissingGlobalValues)) &&
    //           "Referenced metadata not in value map!");
    if (MappedV)
      return mapToMetadata(VM, MD, ValueAsMetadata::get(MappedV), Materializer,
                           Flags);
    return nullptr;
  }

  // Note: this cast precedes the Flags check so we always get its associated
  // assertion.
  const MDNode *Node = cast<MDNode>(MD);

  // If this is a module-level metadata and we know that nothing at the
  // module level is changing, then use an identity mapping.
  if (Flags & RF_NoModuleLevelChanges)
    return mapToSelf(VM, MD, Materializer, Flags);

  // Require resolved nodes whenever metadata might be remapped.
  assert(((Flags & RF_HaveUnmaterializedMetadata) || Node->isResolved()) &&
         "Unexpected unresolved node");

  if (Materializer && Node->isTemporary()) {
    assert(Flags & RF_HaveUnmaterializedMetadata);
    Metadata *TempMD =
        Materializer->mapTemporaryMetadata(const_cast<Metadata *>(MD));
    // If the above callback returned an existing temporary node, use it
    // instead of the current temporary node. This happens when earlier
    // function importing passes already created and saved a temporary
    // metadata node for the same value id.
    if (TempMD) {
      mapToMetadata(VM, MD, TempMD, Materializer, Flags);
      return TempMD;
    }
  }

  if (Node->isDistinct())
    return mapDistinctNode(Node, DistinctWorklist, VM, Flags, TypeMapper,
                           Materializer);

  return mapUniquedNode(Node, DistinctWorklist, VM, Flags, TypeMapper,
                        Materializer);
}

Metadata *llvm::MapMetadata(const Metadata *MD, ValueToValueMapTy &VM,
                            RemapFlags Flags, ValueMapTypeRemapper *TypeMapper,
                            ValueMaterializer *Materializer) {
  SmallVector<MDNode *, 8> DistinctWorklist;
  Metadata *NewMD = MapMetadataImpl(MD, DistinctWorklist, VM, Flags, TypeMapper,
                                    Materializer);

  // When there are no module-level changes, it's possible that the metadata
  // graph has temporaries.  Skip the logic to resolve cycles, since it's
  // unnecessary (and invalid) in that case.
  if (Flags & RF_NoModuleLevelChanges)
    return NewMD;

  // Resolve cycles involving the entry metadata.
  resolveCycles(NewMD, Flags & RF_HaveUnmaterializedMetadata);

  // Remap the operands of distinct MDNodes.
  while (!DistinctWorklist.empty())
    remapOperands(*DistinctWorklist.pop_back_val(), DistinctWorklist, VM, Flags,
                  TypeMapper, Materializer);

  return NewMD;
}

MDNode *llvm::MapMetadata(const MDNode *MD, ValueToValueMapTy &VM,
                          RemapFlags Flags, ValueMapTypeRemapper *TypeMapper,
                          ValueMaterializer *Materializer) {
  return cast<MDNode>(MapMetadata(static_cast<const Metadata *>(MD), VM, Flags,
                                  TypeMapper, Materializer));
}

/// RemapInstruction - Convert the instruction operands from referencing the
/// current values into those specified by VMap.
///
void llvm::RemapInstruction(Instruction *I, ValueToValueMapTy &VMap,
                            RemapFlags Flags, ValueMapTypeRemapper *TypeMapper,
                            ValueMaterializer *Materializer){
  // Remap operands.
  for (User::op_iterator op = I->op_begin(), E = I->op_end(); op != E; ++op) {
    Value *V = MapValue(*op, VMap, Flags, TypeMapper, Materializer);
    // If we aren't ignoring missing entries, assert that something happened.
    if (V)
      *op = V;
    else
      assert((Flags & RF_IgnoreMissingEntries) &&
             "Referenced value not in value map!");
  }

  // Remap phi nodes' incoming blocks.
  if (PHINode *PN = dyn_cast<PHINode>(I)) {
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
      Value *V = MapValue(PN->getIncomingBlock(i), VMap, Flags);
      // If we aren't ignoring missing entries, assert that something happened.
      if (V)
        PN->setIncomingBlock(i, cast<BasicBlock>(V));
      else
        assert((Flags & RF_IgnoreMissingEntries) &&
               "Referenced block not in value map!");
    }
  }

  // Remap attached metadata.
  SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
  I->getAllMetadata(MDs);
  for (const auto &MI : MDs) {
    MDNode *Old = MI.second;
    MDNode *New = MapMetadata(Old, VMap, Flags, TypeMapper, Materializer);
    if (New != Old)
      I->setMetadata(MI.first, New);
  }
  
  if (!TypeMapper)
    return;

  // If the instruction's type is being remapped, do so now.
  if (auto CS = CallSite(I)) {
    SmallVector<Type *, 3> Tys;
    FunctionType *FTy = CS.getFunctionType();
    Tys.reserve(FTy->getNumParams());
    for (Type *Ty : FTy->params())
      Tys.push_back(TypeMapper->remapType(Ty));
    CS.mutateFunctionType(FunctionType::get(
        TypeMapper->remapType(I->getType()), Tys, FTy->isVarArg()));
    return;
  }
  if (auto *AI = dyn_cast<AllocaInst>(I))
    AI->setAllocatedType(TypeMapper->remapType(AI->getAllocatedType()));
  if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
    GEP->setSourceElementType(
        TypeMapper->remapType(GEP->getSourceElementType()));
    GEP->setResultElementType(
        TypeMapper->remapType(GEP->getResultElementType()));
  }
  I->mutateType(TypeMapper->remapType(I->getType()));
}
