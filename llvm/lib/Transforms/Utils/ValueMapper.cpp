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
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
using namespace llvm;

// Out of line method to get vtable etc for class.
void ValueMapTypeRemapper::anchor() {}
void ValueMaterializer::anchor() {}

Value *llvm::MapValue(const Value *V, ValueToValueMapTy &VM, RemapFlags Flags,
                      ValueMapTypeRemapper *TypeMapper,
                      ValueMaterializer *Materializer) {
  ValueToValueMapTy::iterator I = VM.find(V);
  
  // If the value already exists in the map, use it.
  if (I != VM.end() && I->second) return I->second;
  
  // If we have a materializer and it can materialize a value, use that.
  if (Materializer) {
    if (Value *NewV = Materializer->materializeValueFor(const_cast<Value*>(V)))
      return VM[V] = NewV;
  }

  // Global values do not need to be seeded into the VM if they
  // are using the identity mapping.
  if (isa<GlobalValue>(V))
    return VM[V] = const_cast<Value*>(V);
  
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
    //    assert(MappedMD && "Referenced metadata value not in value map");
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
  
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C))
    return VM[V] = CE->getWithOperands(Ops, NewTy);
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
                     Metadata *Val) {
  VM.MD()[Key].reset(Val);
  return Val;
}

static Metadata *mapToSelf(ValueToValueMapTy &VM, const Metadata *MD) {
  return mapToMetadata(VM, MD, const_cast<Metadata *>(MD));
}

static Metadata *MapMetadataImpl(const Metadata *MD,
                                 SmallVectorImpl<UniquableMDNode *> &Cycles,
                                 ValueToValueMapTy &VM, RemapFlags Flags,
                                 ValueMapTypeRemapper *TypeMapper,
                                 ValueMaterializer *Materializer);

static Metadata *mapMetadataOp(Metadata *Op,
                               SmallVectorImpl<UniquableMDNode *> &Cycles,
                               ValueToValueMapTy &VM, RemapFlags Flags,
                               ValueMapTypeRemapper *TypeMapper,
                               ValueMaterializer *Materializer) {
  if (!Op)
    return nullptr;
  if (Metadata *MappedOp =
          MapMetadataImpl(Op, Cycles, VM, Flags, TypeMapper, Materializer))
    return MappedOp;
  // Use identity map if MappedOp is null and we can ignore missing entries.
  if (Flags & RF_IgnoreMissingEntries)
    return Op;

  // FIXME: This assert crashes during bootstrap, but I think it should be
  // correct.  For now, just match behaviour from before the metadata/value
  // split.
  //
  //    llvm_unreachable("Referenced metadata not in value map!");
  return nullptr;
}

static Metadata *cloneMDTuple(const MDTuple *Node,
                              SmallVectorImpl<UniquableMDNode *> &Cycles,
                              ValueToValueMapTy &VM, RemapFlags Flags,
                              ValueMapTypeRemapper *TypeMapper,
                              ValueMaterializer *Materializer,
                              bool IsDistinct) {
  // Distinct MDTuples have their own code path.
  assert(!IsDistinct && "Unexpected distinct tuple");
  (void)IsDistinct;

  SmallVector<Metadata *, 4> Elts;
  Elts.reserve(Node->getNumOperands());
  for (unsigned I = 0, E = Node->getNumOperands(); I != E; ++I)
    Elts.push_back(mapMetadataOp(Node->getOperand(I), Cycles, VM, Flags,
                                 TypeMapper, Materializer));

  return MDTuple::get(Node->getContext(), Elts);
}

static Metadata *cloneMDLocation(const MDLocation *Node,
                                 SmallVectorImpl<UniquableMDNode *> &Cycles,
                                 ValueToValueMapTy &VM, RemapFlags Flags,
                                 ValueMapTypeRemapper *TypeMapper,
                                 ValueMaterializer *Materializer,
                                 bool IsDistinct) {
  return (IsDistinct ? MDLocation::getDistinct : MDLocation::get)(
      Node->getContext(), Node->getLine(), Node->getColumn(),
      mapMetadataOp(Node->getScope(), Cycles, VM, Flags, TypeMapper,
                    Materializer),
      mapMetadataOp(Node->getInlinedAt(), Cycles, VM, Flags, TypeMapper,
                    Materializer));
}

static Metadata *cloneMDNode(const UniquableMDNode *Node,
                             SmallVectorImpl<UniquableMDNode *> &Cycles,
                             ValueToValueMapTy &VM, RemapFlags Flags,
                             ValueMapTypeRemapper *TypeMapper,
                             ValueMaterializer *Materializer, bool IsDistinct) {
  switch (Node->getMetadataID()) {
  default:
    llvm_unreachable("Invalid UniquableMDNode subclass");
#define HANDLE_UNIQUABLE_LEAF(CLASS)                                           \
  case Metadata::CLASS##Kind:                                                  \
    return clone##CLASS(cast<CLASS>(Node), Cycles, VM, Flags, TypeMapper,      \
                        Materializer, IsDistinct);
#include "llvm/IR/Metadata.def"
  }
}

static void
trackCyclesUnderDistinct(const UniquableMDNode *Node,
                         SmallVectorImpl<UniquableMDNode *> &Cycles) {
  // Track any cycles beneath this node.
  for (Metadata *Op : Node->operands())
    if (auto *N = dyn_cast_or_null<UniquableMDNode>(Op))
      if (!N->isResolved())
        Cycles.push_back(N);
}

/// \brief Map a distinct MDNode.
///
/// Distinct nodes are not uniqued, so they must always recreated.
static Metadata *mapDistinctNode(const UniquableMDNode *Node,
                                 SmallVectorImpl<UniquableMDNode *> &Cycles,
                                 ValueToValueMapTy &VM, RemapFlags Flags,
                                 ValueMapTypeRemapper *TypeMapper,
                                 ValueMaterializer *Materializer) {
  assert(Node->isDistinct() && "Expected distinct node");

  // Optimization for MDTuples.
  if (isa<MDTuple>(Node)) {
    // Create the node first so it's available for cyclical references.
    SmallVector<Metadata *, 4> EmptyOps(Node->getNumOperands());
    MDTuple *NewMD = MDTuple::getDistinct(Node->getContext(), EmptyOps);
    mapToMetadata(VM, Node, NewMD);

    // Fix the operands.
    for (unsigned I = 0, E = Node->getNumOperands(); I != E; ++I)
      NewMD->replaceOperandWith(I,
                                mapMetadataOp(Node->getOperand(I), Cycles, VM,
                                              Flags, TypeMapper, Materializer));

    trackCyclesUnderDistinct(NewMD, Cycles);
    return NewMD;
  }

  // In general we need a dummy node, since whether the operands are null can
  // affect the size of the node.
  std::unique_ptr<MDNodeFwdDecl> Dummy(
      MDNode::getTemporary(Node->getContext(), None));
  mapToMetadata(VM, Node, Dummy.get());
  auto *NewMD = cast<UniquableMDNode>(cloneMDNode(Node, Cycles, VM, Flags,
                                                  TypeMapper, Materializer,
                                                  /* IsDistinct */ true));
  Dummy->replaceAllUsesWith(NewMD);
  trackCyclesUnderDistinct(NewMD, Cycles);
  return mapToMetadata(VM, Node, NewMD);
}

/// \brief Check whether a uniqued node needs to be remapped.
///
/// Check whether a uniqued node needs to be remapped (due to any operands
/// changing).
static bool shouldRemapUniquedNode(const UniquableMDNode *Node,
                                   SmallVectorImpl<UniquableMDNode *> &Cycles,
                                   ValueToValueMapTy &VM, RemapFlags Flags,
                                   ValueMapTypeRemapper *TypeMapper,
                                   ValueMaterializer *Materializer) {
  // Check all operands to see if any need to be remapped.
  for (unsigned I = 0, E = Node->getNumOperands(); I != E; ++I) {
    Metadata *Op = Node->getOperand(I);
    if (Op != mapMetadataOp(Op, Cycles, VM, Flags, TypeMapper, Materializer))
      return true;
  }
  return false;
}

/// \brief Map a uniqued MDNode.
///
/// Uniqued nodes may not need to be recreated (they may map to themselves).
static Metadata *mapUniquedNode(const UniquableMDNode *Node,
                                SmallVectorImpl<UniquableMDNode *> &Cycles,
                                ValueToValueMapTy &VM, RemapFlags Flags,
                                ValueMapTypeRemapper *TypeMapper,
                                ValueMaterializer *Materializer) {
  assert(!Node->isDistinct() && "Expected uniqued node");

  // Create a dummy node in case we have a metadata cycle.
  MDNodeFwdDecl *Dummy = MDNode::getTemporary(Node->getContext(), None);
  mapToMetadata(VM, Node, Dummy);

  // Check all operands to see if any need to be remapped.
  if (!shouldRemapUniquedNode(Node, Cycles, VM, Flags, TypeMapper,
                              Materializer)) {
    // Use an identity mapping.
    mapToSelf(VM, Node);
    MDNode::deleteTemporary(Dummy);
    return const_cast<Metadata *>(static_cast<const Metadata *>(Node));
  }

  // At least one operand needs remapping.
  Metadata *NewMD =
      cloneMDNode(Node, Cycles, VM, Flags, TypeMapper, Materializer,
                  /* IsDistinct */ false);
  Dummy->replaceAllUsesWith(NewMD);
  MDNode::deleteTemporary(Dummy);
  return mapToMetadata(VM, Node, NewMD);
}

static Metadata *MapMetadataImpl(const Metadata *MD,
                                 SmallVectorImpl<UniquableMDNode *> &Cycles,
                                 ValueToValueMapTy &VM, RemapFlags Flags,
                                 ValueMapTypeRemapper *TypeMapper,
                                 ValueMaterializer *Materializer) {
  // If the value already exists in the map, use it.
  if (Metadata *NewMD = VM.MD().lookup(MD).get())
    return NewMD;

  if (isa<MDString>(MD))
    return mapToSelf(VM, MD);

  if (isa<ConstantAsMetadata>(MD))
    if ((Flags & RF_NoModuleLevelChanges))
      return mapToSelf(VM, MD);

  if (const auto *VMD = dyn_cast<ValueAsMetadata>(MD)) {
    Value *MappedV =
        MapValue(VMD->getValue(), VM, Flags, TypeMapper, Materializer);
    if (VMD->getValue() == MappedV ||
        (!MappedV && (Flags & RF_IgnoreMissingEntries)))
      return mapToSelf(VM, MD);

    // FIXME: This assert crashes during bootstrap, but I think it should be
    // correct.  For now, just match behaviour from before the metadata/value
    // split.
    //
    //    assert(MappedV && "Referenced metadata not in value map!");
    if (MappedV)
      return mapToMetadata(VM, MD, ValueAsMetadata::get(MappedV));
    return nullptr;
  }

  const UniquableMDNode *Node = cast<UniquableMDNode>(MD);
  assert(Node->isResolved() && "Unexpected unresolved node");

  // If this is a module-level metadata and we know that nothing at the
  // module level is changing, then use an identity mapping.
  if (Flags & RF_NoModuleLevelChanges)
    return mapToSelf(VM, MD);

  if (Node->isDistinct())
    return mapDistinctNode(Node, Cycles, VM, Flags, TypeMapper, Materializer);

  return mapUniquedNode(Node, Cycles, VM, Flags, TypeMapper, Materializer);
}

Metadata *llvm::MapMetadata(const Metadata *MD, ValueToValueMapTy &VM,
                            RemapFlags Flags, ValueMapTypeRemapper *TypeMapper,
                            ValueMaterializer *Materializer) {
  SmallVector<UniquableMDNode *, 8> Cycles;
  Metadata *NewMD =
      MapMetadataImpl(MD, Cycles, VM, Flags, TypeMapper, Materializer);

  // Resolve cycles underneath MD.
  if (NewMD && NewMD != MD) {
    if (auto *N = dyn_cast<UniquableMDNode>(NewMD))
      N->resolveCycles();

    for (UniquableMDNode *N : Cycles)
      N->resolveCycles();
  } else {
    // Shouldn't get unresolved cycles if nothing was remapped.
    assert(Cycles.empty() && "Expected no unresolved cycles");
  }

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
  for (SmallVectorImpl<std::pair<unsigned, MDNode *>>::iterator
           MI = MDs.begin(),
           ME = MDs.end();
       MI != ME; ++MI) {
    MDNode *Old = MI->second;
    MDNode *New = MapMetadata(Old, VMap, Flags, TypeMapper, Materializer);
    if (New != Old)
      I->setMetadata(MI->first, New);
  }
  
  // If the instruction's type is being remapped, do so now.
  if (TypeMapper)
    I->mutateType(TypeMapper->remapType(I->getType()));
}
