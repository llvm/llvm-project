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

namespace {

/// A GlobalValue whose initializer needs to be materialized.
struct DelayedGlobalValueInit {
  GlobalValue *Old;
  GlobalValue *New;
  DelayedGlobalValueInit(const GlobalValue *Old, GlobalValue *New)
      : Old(const_cast<GlobalValue *>(Old)), New(New) {}
};

/// A basic block used in a BlockAddress whose function body is not yet
/// materialized.
struct DelayedBasicBlock {
  BasicBlock *OldBB;
  std::unique_ptr<BasicBlock> TempBB;

  // Explicit move for MSVC.
  DelayedBasicBlock(DelayedBasicBlock &&X)
      : OldBB(std::move(X.OldBB)), TempBB(std::move(X.TempBB)) {}
  DelayedBasicBlock &operator=(DelayedBasicBlock &&X) {
    OldBB = std::move(X.OldBB);
    TempBB = std::move(X.TempBB);
    return *this;
  }

  DelayedBasicBlock(const BlockAddress &Old)
      : OldBB(Old.getBasicBlock()),
        TempBB(BasicBlock::Create(Old.getContext())) {}
};

class Mapper {
  ValueToValueMapTy &VM;
  RemapFlags Flags;
  ValueMapTypeRemapper *TypeMapper;
  ValueMaterializer *Materializer;

  SmallVector<DelayedGlobalValueInit, 8> DelayedInits;
  SmallVector<DelayedBasicBlock, 1> DelayedBBs;
  SmallVector<MDNode *, 8> DistinctWorklist;

public:
  Mapper(ValueToValueMapTy &VM, RemapFlags Flags,
         ValueMapTypeRemapper *TypeMapper, ValueMaterializer *Materializer)
      : VM(VM), Flags(Flags), TypeMapper(TypeMapper),
        Materializer(Materializer) {}

  ~Mapper();

  Value *mapValue(const Value *V);

  /// Map metadata.
  ///
  /// Find the mapping for MD.  Guarantees that the return will be resolved
  /// (not an MDNode, or MDNode::isResolved() returns true).
  Metadata *mapMetadata(const Metadata *MD);

private:
  Value *mapBlockAddress(const BlockAddress &BA);

  /// Map metadata helper.
  ///
  /// Co-recursively finds the mapping for MD.  If this returns an MDNode, it's
  /// possible that MDNode::isResolved() will return false.
  Metadata *mapMetadataImpl(const Metadata *MD);
  Metadata *mapMetadataOp(Metadata *Op);

  /// Map metadata that doesn't require visiting operands.
  Optional<Metadata *> mapSimpleMetadata(const Metadata *MD);

  /// Remap the operands of an MDNode.
  ///
  /// If \c Node is temporary, uniquing cycles are ignored.  If \c Node is
  /// distinct, uniquing cycles are resolved as they're found.
  ///
  /// \pre \c Node.isDistinct() or \c Node.isTemporary().
  bool remapOperands(MDNode &Node);

  /// Map a distinct MDNode.
  ///
  /// Whether distinct nodes change is independent of their operands.  If \a
  /// RF_MoveDistinctMDs, then they are reused, and their operands remapped in
  /// place; effectively, they're moved from one graph to another.  Otherwise,
  /// they're cloned/duplicated, and the new copy's operands are remapped.
  Metadata *mapDistinctNode(const MDNode *Node);

  /// Map a uniqued MDNode.
  ///
  /// Uniqued nodes may not need to be recreated (they may map to themselves).
  Metadata *mapUniquedNode(const MDNode *Node);

  Metadata *mapToMetadata(const Metadata *Key, Metadata *Val);
  Metadata *mapToSelf(const Metadata *MD);
};

} // end namespace

Value *llvm::MapValue(const Value *V, ValueToValueMapTy &VM, RemapFlags Flags,
                      ValueMapTypeRemapper *TypeMapper,
                      ValueMaterializer *Materializer) {
  return Mapper(VM, Flags, TypeMapper, Materializer).mapValue(V);
}

Value *Mapper::mapValue(const Value *V) {
  ValueToValueMapTy::iterator I = VM.find(V);
  
  // If the value already exists in the map, use it.
  if (I != VM.end() && I->second) return I->second;
  
  // If we have a materializer and it can materialize a value, use that.
  if (Materializer) {
    if (Value *NewV =
            Materializer->materializeDeclFor(const_cast<Value *>(V))) {
      VM[V] = NewV;
      if (auto *NewGV = dyn_cast<GlobalValue>(NewV))
        DelayedInits.push_back(
            DelayedGlobalValueInit(cast<GlobalValue>(V), NewGV));
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

    auto *MappedMD = mapMetadata(MD);
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

  if (BlockAddress *BA = dyn_cast<BlockAddress>(C))
    return mapBlockAddress(*BA);

  // Otherwise, we have some other constant to remap.  Start by checking to see
  // if all operands have an identity remapping.
  unsigned OpNo = 0, NumOperands = C->getNumOperands();
  Value *Mapped = nullptr;
  for (; OpNo != NumOperands; ++OpNo) {
    Value *Op = C->getOperand(OpNo);
    Mapped = mapValue(Op);
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
      Ops.push_back(cast<Constant>(mapValue(C->getOperand(OpNo))));
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

Value *Mapper::mapBlockAddress(const BlockAddress &BA) {
  Function *F = cast<Function>(mapValue(BA.getFunction()));

  // F may not have materialized its initializer.  In that case, create a
  // dummy basic block for now, and replace it once we've materialized all
  // the initializers.
  BasicBlock *BB;
  if (F->isDeclaration()) {
    BB = cast_or_null<BasicBlock>(mapValue(BA.getBasicBlock()));
  } else {
    DelayedBBs.push_back(DelayedBasicBlock(BA));
    BB = DelayedBBs.back().TempBB.get();
  }

  return VM[&BA] = BlockAddress::get(F, BB ? BB : BA.getBasicBlock());
}

Metadata *Mapper::mapToMetadata(const Metadata *Key, Metadata *Val) {
  VM.MD()[Key].reset(Val);
  return Val;
}

Metadata *Mapper::mapToSelf(const Metadata *MD) {
  return mapToMetadata(MD, const_cast<Metadata *>(MD));
}

Metadata *Mapper::mapMetadataOp(Metadata *Op) {
  if (!Op)
    return nullptr;

  if (Metadata *MappedOp = mapMetadataImpl(Op))
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
static void resolveCycles(Metadata *MD) {
  if (auto *N = dyn_cast_or_null<MDNode>(MD))
    if (!N->isResolved())
      N->resolveCycles();
}

bool Mapper::remapOperands(MDNode &Node) {
  assert(!Node.isUniqued() && "Expected temporary or distinct node");
  const bool IsDistinct = Node.isDistinct();

  bool AnyChanged = false;
  for (unsigned I = 0, E = Node.getNumOperands(); I != E; ++I) {
    Metadata *Old = Node.getOperand(I);
    Metadata *New = mapMetadataOp(Old);
    if (Old != New) {
      AnyChanged = true;
      Node.replaceOperandWith(I, New);

      // Resolve uniquing cycles underneath distinct nodes on the fly so they
      // don't infect later operands.
      if (IsDistinct)
        resolveCycles(New);
    }
  }

  return AnyChanged;
}

Metadata *Mapper::mapDistinctNode(const MDNode *Node) {
  assert(Node->isDistinct() && "Expected distinct node");

  MDNode *NewMD;
  if (Flags & RF_MoveDistinctMDs)
    NewMD = const_cast<MDNode *>(Node);
  else
    NewMD = MDNode::replaceWithDistinct(Node->clone());

  // Remap operands later.
  DistinctWorklist.push_back(NewMD);
  return mapToMetadata(Node, NewMD);
}

Metadata *Mapper::mapUniquedNode(const MDNode *Node) {
  assert(Node->isUniqued() && "Expected uniqued node");

  // Create a temporary node and map it upfront in case we have a uniquing
  // cycle.  If necessary, this mapping will get updated by RAUW logic before
  // returning.
  auto ClonedMD = Node->clone();
  mapToMetadata(Node, ClonedMD.get());
  if (!remapOperands(*ClonedMD)) {
    // No operands changed, so use the original.
    ClonedMD->replaceAllUsesWith(const_cast<MDNode *>(Node));
    return const_cast<MDNode *>(Node);
  }

  // Uniquify the cloned node.
  return MDNode::replaceWithUniqued(std::move(ClonedMD));
}

Optional<Metadata *> Mapper::mapSimpleMetadata(const Metadata *MD) {
  // If the value already exists in the map, use it.
  if (Optional<Metadata *> NewMD = VM.getMappedMD(MD))
    return *NewMD;

  if (isa<MDString>(MD))
    return mapToSelf(MD);

  if (isa<ConstantAsMetadata>(MD))
    if ((Flags & RF_NoModuleLevelChanges))
      return mapToSelf(MD);

  if (const auto *VMD = dyn_cast<ValueAsMetadata>(MD)) {
    Value *MappedV = mapValue(VMD->getValue());
    if (VMD->getValue() == MappedV ||
        (!MappedV && (Flags & RF_IgnoreMissingEntries)))
      return mapToSelf(MD);

    // FIXME: This assert crashes during bootstrap, but I think it should be
    // correct.  For now, just match behaviour from before the metadata/value
    // split.
    //
    //    assert((MappedV || (Flags & RF_NullMapMissingGlobalValues)) &&
    //           "Referenced metadata not in value map!");
    if (MappedV)
      return mapToMetadata(MD, ValueAsMetadata::get(MappedV));
    return nullptr;
  }

  assert(isa<MDNode>(MD) && "Expected a metadata node");

  // If this is a module-level metadata and we know that nothing at the
  // module level is changing, then use an identity mapping.
  if (Flags & RF_NoModuleLevelChanges)
    return mapToSelf(MD);

  return None;
}

Metadata *Mapper::mapMetadataImpl(const Metadata *MD) {
  if (Optional<Metadata *> NewMD = mapSimpleMetadata(MD))
    return *NewMD;

  // Require resolved nodes whenever metadata might be remapped.
  auto *Node = cast<MDNode>(MD);
  assert(Node->isResolved() && "Unexpected unresolved node");

  if (Node->isDistinct())
    return mapDistinctNode(Node);

  return mapUniquedNode(Node);
}

Metadata *llvm::MapMetadata(const Metadata *MD, ValueToValueMapTy &VM,
                            RemapFlags Flags, ValueMapTypeRemapper *TypeMapper,
                            ValueMaterializer *Materializer) {
  return Mapper(VM, Flags, TypeMapper, Materializer).mapMetadata(MD);
}

Metadata *Mapper::mapMetadata(const Metadata *MD) {
  Metadata *NewMD = mapMetadataImpl(MD);

  // When there are no module-level changes, it's possible that the metadata
  // graph has temporaries.  Skip the logic to resolve cycles, since it's
  // unnecessary (and invalid) in that case.
  if (Flags & RF_NoModuleLevelChanges)
    return NewMD;

  // Resolve cycles involving the entry metadata.
  resolveCycles(NewMD);

  return NewMD;
}

Mapper::~Mapper() {
  // Remap the operands of distinct MDNodes.
  while (!DistinctWorklist.empty())
    remapOperands(*DistinctWorklist.pop_back_val());

  // Materialize global initializers.
  while (!DelayedInits.empty()) {
    auto Init = DelayedInits.pop_back_val();
    Materializer->materializeInitFor(Init.New, Init.Old);
  }

  // Process block addresses delayed until global inits.
  while (!DelayedBBs.empty()) {
    DelayedBasicBlock DBB = DelayedBBs.pop_back_val();
    BasicBlock *BB = cast_or_null<BasicBlock>(mapValue(DBB.OldBB));
    DBB.TempBB->replaceAllUsesWith(BB ? BB : DBB.OldBB);
  }

  // We don't expect any of these to grow after clearing.
  assert(DistinctWorklist.empty());
  assert(DelayedInits.empty());
  assert(DelayedBBs.empty());
}

MDNode *llvm::MapMetadata(const MDNode *MD, ValueToValueMapTy &VM,
                          RemapFlags Flags, ValueMapTypeRemapper *TypeMapper,
                          ValueMaterializer *Materializer) {
  return cast_or_null<MDNode>(MapMetadata(static_cast<const Metadata *>(MD), VM,
                                          Flags, TypeMapper, Materializer));
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
