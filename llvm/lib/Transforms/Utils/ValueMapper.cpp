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

class MDNodeMapper;
class Mapper {
  friend class MDNodeMapper;

  ValueToValueMapTy &VM;
  RemapFlags Flags;
  ValueMapTypeRemapper *TypeMapper;
  ValueMaterializer *Materializer;

  SmallVector<DelayedGlobalValueInit, 8> DelayedInits;
  SmallVector<DelayedBasicBlock, 1> DelayedBBs;

public:
  Mapper(ValueToValueMapTy &VM, RemapFlags Flags,
         ValueMapTypeRemapper *TypeMapper, ValueMaterializer *Materializer)
      : VM(VM), Flags(Flags), TypeMapper(TypeMapper),
        Materializer(Materializer) {}

  ~Mapper();

  Value *mapValue(const Value *V);
  void remapInstruction(Instruction *I);
  void remapFunction(Function &F);

  /// Map metadata.
  ///
  /// Find the mapping for MD.  Guarantees that the return will be resolved
  /// (not an MDNode, or MDNode::isResolved() returns true).
  Metadata *mapMetadata(const Metadata *MD);

  // Map LocalAsMetadata, which never gets memoized.
  //
  // If the referenced local is not mapped, the principled return is nullptr.
  // However, optimization passes sometimes move metadata operands *before* the
  // SSA values they reference.  To prevent crashes in \a RemapInstruction(),
  // return "!{}" when RF_IgnoreMissingLocals is not set.
  //
  // \note Adding a mapping for LocalAsMetadata is unsupported.  Add a mapping
  // to the value map for the SSA value in question instead.
  //
  // FIXME: Once we have a verifier check for forward references to SSA values
  // through metadata operands, always return nullptr on unmapped locals.
  Metadata *mapLocalAsMetadata(const LocalAsMetadata &LAM);

private:
  Value *mapBlockAddress(const BlockAddress &BA);

  /// Map metadata that doesn't require visiting operands.
  Optional<Metadata *> mapSimpleMetadata(const Metadata *MD);

  Metadata *mapToMetadata(const Metadata *Key, Metadata *Val);
  Metadata *mapToSelf(const Metadata *MD);
};

class MDNodeMapper {
  Mapper &M;

  struct Data {
    bool HasChangedOps = false;
    bool HasChangedAddress = false;
    unsigned ID = ~0u;
    TempMDNode Placeholder;

    Data() {}
    Data(Data &&X)
        : HasChangedOps(std::move(X.HasChangedOps)),
          HasChangedAddress(std::move(X.HasChangedAddress)),
          ID(std::move(X.ID)), Placeholder(std::move(X.Placeholder)) {}
    Data &operator=(Data &&X) {
      HasChangedOps = std::move(X.HasChangedOps);
      HasChangedAddress = std::move(X.HasChangedAddress);
      ID = std::move(X.ID);
      Placeholder = std::move(X.Placeholder);
      return *this;
    }
  };

  SmallDenseMap<const Metadata *, Data, 32> Info;
  SmallVector<std::pair<MDNode *, bool>, 16> Worklist;
  SmallVector<MDNode *, 16> POT;

public:
  MDNodeMapper(Mapper &M) : M(M) {}

  /// Map a metadata node (and its transitive operands).
  ///
  /// This is the only entry point into MDNodeMapper.  It works as follows:
  ///
  ///  1. \a createPOT(): use a worklist to perform a post-order traversal of
  ///     the transitively referenced unmapped nodes.
  ///
  ///  2. \a propagateChangedOperands(): track which nodes will change
  ///     operands, and which will have new addresses in the mapped scheme.
  ///     Propagate the changes through the POT until fixed point, to pick up
  ///     uniquing cycles that need to change.
  ///
  ///  3. \a mapDistinctNodes(): map all the distinct nodes without touching
  ///     their operands.  If RF_MoveDistinctMetadata, they get mapped to
  ///     themselves; otherwise, they get mapped to clones.
  ///
  ///  4. \a mapUniquedNodes(): map the uniqued nodes (bottom-up), lazily
  ///     creating temporaries for forward references as needed.
  ///
  ///  5. \a remapDistinctOperands(): remap the operands of the distinct nodes.
  Metadata *map(const MDNode &FirstN);

private:
  /// Return \c true as long as there's work to do.
  bool hasWork() const { return !Worklist.empty(); }

  /// Get the current node in the worklist.
  MDNode &getCurrentNode() const { return *Worklist.back().first; }

  /// Push a node onto the worklist.
  ///
  /// Adds \c N to \a Worklist and \a Info, unless it's already inserted.  If
  /// \c N.isDistinct(), \a Data::HasChangedAddress will be set based on \a
  /// RF_MoveDistinctMDs.
  ///
  /// Returns the data for the node.
  ///
  /// \post Data::HasChangedAddress iff !RF_MoveDistinctMDs && N.isDistinct().
  /// \post Worklist.back().first == &N.
  /// \post Worklist.back().second == false.
  Data &push(const MDNode &N);

  /// Map a node operand, and return true if it changes.
  ///
  /// \post getMappedOp(Op) does not return None.
  bool mapOperand(const Metadata *Op);

  /// Get a previously mapped node.
  Optional<Metadata *> getMappedOp(const Metadata *Op) const;

  /// Try to pop a node off the worklist and store it in POT.
  ///
  /// Returns \c true if it popped; \c false if its operands need to be
  /// visited.
  ///
  /// \post If Worklist.back().second == false: Worklist.back().second == true.
  /// \post Else: Worklist.back() has been popped off and added to \a POT.
  bool tryToPop();

  /// Get a forward reference to a node to use as an operand.
  ///
  /// Returns \c Op if it's not changing; otherwise, lazily creates a temporary
  /// node and returns it.
  Metadata &getFwdReference(const Data &D, MDNode &Op);

  /// Create a post-order traversal from the given node.
  ///
  /// This traverses the metadata graph deeply enough to map \c FirstN.  It
  /// uses \a mapOperand() (indirectly, \a Mapper::mapSimplifiedNode()), so any
  /// metadata that has already been mapped will not be part of the POT.
  ///
  /// \post \a POT is a post-order traversal ending with \c FirstN.
  bool createPOT(const MDNode &FirstN);

  /// Propagate changed operands through post-order traversal.
  ///
  /// Until fixed point, iteratively update:
  ///
  ///   - \a Data::HasChangedOps based on \a Data::HasChangedAddress of operands;
  ///   - \a Data::HasChangedAddress based on Data::HasChangedOps.
  ///
  /// This algorithm never changes \a Data::HasChangedAddress for distinct
  /// nodes.
  ///
  /// \post \a POT is a post-order traversal ending with \c FirstN.
  void propagateChangedOperands();

  /// Map all distinct nodes in POT.
  ///
  /// \post \a getMappedOp() returns the correct node for every distinct node.
  void mapDistinctNodes();

  /// Map all uniqued nodes in POT with the correct operands.
  ///
  /// \pre Distinct nodes are mapped (\a mapDistinctNodes() has been called).
  /// \post \a getMappedOp() returns the correct node for every node.
  /// \post \a MDNode::operands() is correct for every uniqued node.
  /// \post \a MDNode::isResolved() returns true for every node.
  void mapUniquedNodes();

  /// Re-map the operands for distinct nodes in POT.
  ///
  /// \pre Distinct nodes are mapped (\a mapDistinctNodes() has been called).
  /// \pre Uniqued nodes are mapped (\a mapUniquedNodes() has been called).
  /// \post \a MDNode::operands() is correct for every distinct node.
  void remapDistinctOperands();

  /// Remap a node's operands.
  ///
  /// Iterate through operands and update them in place using \a getMappedOp()
  /// and \a getFwdReference().
  ///
  /// \pre N.isDistinct() or N.isTemporary().
  /// \pre Distinct nodes are mapped (\a mapDistinctNodes() has been called).
  /// \pre If \c N is distinct, all uniqued nodes are already mapped.
  void remapOperands(const Data &D, MDNode &N);
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
    if (Flags & RF_NullMapMissingGlobalValues)
      return nullptr;
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

    if (auto *LAM = dyn_cast<LocalAsMetadata>(MD)) {
      // Look through to grab the local value.
      if (Value *LV = mapValue(LAM->getValue())) {
        if (V == LAM->getValue())
          return const_cast<Value *>(V);
        return MetadataAsValue::get(V->getContext(), ValueAsMetadata::get(LV));
      }

      // FIXME: always return nullptr once Verifier::verifyDominatesUse()
      // ensures metadata operands only reference defined SSA values.
      return (Flags & RF_IgnoreMissingLocals)
                 ? nullptr
                 : MetadataAsValue::get(V->getContext(),
                                        MDTuple::get(V->getContext(), None));
    }

    // If this is a module-level metadata and we know that nothing at the module
    // level is changing, then use an identity mapping.
    if (Flags & RF_NoModuleLevelChanges)
      return VM[V] = const_cast<Value *>(V);

    // Map the metadata and turn it into a value.
    auto *MappedMD = mapMetadata(MD);
    if (MD == MappedMD)
      return VM[V] = const_cast<Value *>(V);
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
  if (F->empty()) {
    DelayedBBs.push_back(DelayedBasicBlock(BA));
    BB = DelayedBBs.back().TempBB.get();
  } else {
    BB = cast_or_null<BasicBlock>(mapValue(BA.getBasicBlock()));
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

bool MDNodeMapper::mapOperand(const Metadata *Op) {
  if (!Op)
    return false;

  if (Optional<Metadata *> MappedOp = M.mapSimpleMetadata(Op)) {
    assert((isa<MDString>(Op) || M.VM.getMappedMD(Op)) &&
           "Expected result to be memoized");
    return *MappedOp != Op;
  }

  return push(*cast<MDNode>(Op)).HasChangedAddress;
}

Optional<Metadata *> MDNodeMapper::getMappedOp(const Metadata *Op) const {
  if (!Op)
    return nullptr;

  if (Optional<Metadata *> MappedOp = M.VM.getMappedMD(Op))
    return *MappedOp;

  if (isa<MDString>(Op))
    return const_cast<Metadata *>(Op);

  return None;
}

Metadata &MDNodeMapper::getFwdReference(const Data &D, MDNode &Op) {
  auto Where = Info.find(&Op);
  assert(Where != Info.end() && "Expected a valid reference");

  auto &OpD = Where->second;
  assert(OpD.ID > D.ID && "Expected a forward reference");

  if (!OpD.HasChangedAddress)
    return Op;

  // Lazily construct a temporary node.
  if (!OpD.Placeholder)
    OpD.Placeholder = Op.clone();

  return *OpD.Placeholder;
}

void MDNodeMapper::remapOperands(const Data &D, MDNode &N) {
  for (unsigned I = 0, E = N.getNumOperands(); I != E; ++I) {
    Metadata *Old = N.getOperand(I);
    Metadata *New;
    if (Optional<Metadata *> MappedOp = getMappedOp(Old)){
      New = *MappedOp;
    } else {
      assert(!N.isDistinct() &&
             "Expected all nodes to be pre-mapped for distinct operands");
      MDNode &OldN = *cast<MDNode>(Old);
      assert(!OldN.isDistinct() && "Expected distinct nodes to be pre-mapped");
      New = &getFwdReference(D, OldN);
    }

    if (Old != New)
      N.replaceOperandWith(I, New);
  }
}

MDNodeMapper::Data &MDNodeMapper::push(const MDNode &N) {
  auto Insertion = Info.insert(std::make_pair(&N, Data()));
  auto &D = Insertion.first->second;
  if (!Insertion.second)
    return D;

  // Add to the worklist; check for distinct nodes that are required to be
  // copied.
  Worklist.push_back(std::make_pair(&const_cast<MDNode &>(N), false));
  D.HasChangedAddress = !(M.Flags & RF_MoveDistinctMDs) && N.isDistinct();
  return D;
}

bool MDNodeMapper::tryToPop() {
  if (!Worklist.back().second) {
    Worklist.back().second = true;
    return false;
  }

  MDNode *N = Worklist.pop_back_val().first;
  Info[N].ID = POT.size();
  POT.push_back(N);
  return true;
}

bool MDNodeMapper::createPOT(const MDNode &FirstN) {
  bool AnyChanges = false;

  // Do a traversal of the unmapped subgraph, tracking whether operands change.
  // In some cases, these changes will propagate naturally, but
  // propagateChangedOperands() catches the general case.
  AnyChanges |= push(FirstN).HasChangedAddress;
  while (hasWork()) {
    if (tryToPop())
      continue;

    MDNode &N = getCurrentNode();
    bool LocalChanges = false;
    for (const Metadata *Op : N.operands())
      LocalChanges |= mapOperand(Op);

    if (!LocalChanges)
      continue;

    AnyChanges = true;
    auto &D = Info[&N];
    D.HasChangedOps = true;

    // Uniqued nodes change address when operands change.
    if (!N.isDistinct())
      D.HasChangedAddress = true;
  }
  return AnyChanges;
}

void MDNodeMapper::propagateChangedOperands() {
  bool AnyChangedAddresses;
  do {
    AnyChangedAddresses = false;
    for (MDNode *N : POT) {
      auto &NI = Info[N];
      if (NI.HasChangedOps)
        continue;

      if (!llvm::any_of(N->operands(), [&](const Metadata *Op) {
            auto Where = Info.find(Op);
            return Where != Info.end() && Where->second.HasChangedAddress;
          }))
        continue;

      NI.HasChangedOps = true;
      if (!N->isDistinct()) {
        NI.HasChangedAddress = true;
        AnyChangedAddresses = true;
      }
    }
  } while (AnyChangedAddresses);
}

void MDNodeMapper::mapDistinctNodes() {
  // Map all the distinct nodes in POT.
  for (MDNode *N : POT) {
    if (!N->isDistinct())
      continue;

    if (M.Flags & RF_MoveDistinctMDs)
      M.mapToSelf(N);
    else
      M.mapToMetadata(N, MDNode::replaceWithDistinct(N->clone()));
  }
}

void MDNodeMapper::mapUniquedNodes() {
  // Construct uniqued nodes, building forward references as necessary.
  SmallVector<MDNode *, 16> CyclicNodes;
  for (auto *N : POT) {
    if (N->isDistinct())
      continue;

    auto &D = Info[N];
    assert(D.HasChangedAddress == D.HasChangedOps &&
           "Uniqued nodes should change address iff ops change");
    if (!D.HasChangedAddress) {
      M.mapToSelf(N);
      continue;
    }

    TempMDNode ClonedN = D.Placeholder ? std::move(D.Placeholder) : N->clone();
    remapOperands(D, *ClonedN);
    CyclicNodes.push_back(MDNode::replaceWithUniqued(std::move(ClonedN)));
    M.mapToMetadata(N, CyclicNodes.back());
  }

  // Resolve cycles.
  for (auto *N : CyclicNodes)
    if (!N->isResolved())
      N->resolveCycles();
}

void MDNodeMapper::remapDistinctOperands() {
  for (auto *N : POT) {
    if (!N->isDistinct())
      continue;

    auto &D = Info[N];
    if (!D.HasChangedOps)
      continue;

    assert(D.HasChangedAddress == !bool(M.Flags & RF_MoveDistinctMDs) &&
           "Distinct nodes should change address iff they cannot be moved");
    remapOperands(D, D.HasChangedAddress ? *cast<MDNode>(*getMappedOp(N)) : *N);
  }
}

Metadata *MDNodeMapper::map(const MDNode &FirstN) {
  assert(!(M.Flags & RF_NoModuleLevelChanges) &&
         "MDNodeMapper::map assumes module-level changes");
  assert(POT.empty() && "MDNodeMapper::map is not re-entrant");

  // Require resolved nodes whenever metadata might be remapped.
  assert(FirstN.isResolved() && "Unexpected unresolved node");

  // Return early if nothing at all changed.
  if (!createPOT(FirstN)) {
    for (const MDNode *N : POT)
      M.mapToSelf(N);
    return &const_cast<MDNode &>(FirstN);
  }

  propagateChangedOperands();
  mapDistinctNodes();
  mapUniquedNodes();
  remapDistinctOperands();

  // Return the original node, remapped.
  return *getMappedOp(&FirstN);
}

Optional<Metadata *> Mapper::mapSimpleMetadata(const Metadata *MD) {
  // If the value already exists in the map, use it.
  if (Optional<Metadata *> NewMD = VM.getMappedMD(MD))
    return *NewMD;

  if (isa<MDString>(MD))
    return const_cast<Metadata *>(MD);

  // This is a module-level metadata.  If nothing at the module level is
  // changing, use an identity mapping.
  if ((Flags & RF_NoModuleLevelChanges))
    return const_cast<Metadata *>(MD);

  if (auto *CMD = dyn_cast<ConstantAsMetadata>(MD)) {
    // Disallow recursion into metadata mapping through mapValue.
    VM.disableMapMetadata();
    Value *MappedV = mapValue(CMD->getValue());
    VM.enableMapMetadata();

    if (CMD->getValue() == MappedV)
      return mapToSelf(MD);

    return mapToMetadata(MD, MappedV ? ValueAsMetadata::get(MappedV) : nullptr);
  }

  assert(isa<MDNode>(MD) && "Expected a metadata node");

  return None;
}

Metadata *llvm::MapMetadata(const Metadata *MD, ValueToValueMapTy &VM,
                            RemapFlags Flags, ValueMapTypeRemapper *TypeMapper,
                            ValueMaterializer *Materializer) {
  return Mapper(VM, Flags, TypeMapper, Materializer).mapMetadata(MD);
}

Metadata *Mapper::mapLocalAsMetadata(const LocalAsMetadata &LAM) {
  // Lookup the mapping for the value itself, and return the appropriate
  // metadata.
  if (Value *V = mapValue(LAM.getValue())) {
    if (V == LAM.getValue())
      return const_cast<LocalAsMetadata *>(&LAM);
    return ValueAsMetadata::get(V);
  }

  // FIXME: always return nullptr once Verifier::verifyDominatesUse() ensures
  // metadata operands only reference defined SSA values.
  return (Flags & RF_IgnoreMissingLocals)
             ? nullptr
             : MDTuple::get(LAM.getContext(), None);
}

Metadata *Mapper::mapMetadata(const Metadata *MD) {
  assert(MD && "Expected valid metadata");
  assert(!isa<LocalAsMetadata>(MD) && "Unexpected local metadata");

  if (Optional<Metadata *> NewMD = mapSimpleMetadata(MD))
    return *NewMD;

  return MDNodeMapper(*this).map(*cast<MDNode>(MD));
}

Mapper::~Mapper() {
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

  // We don't expect these to grow after clearing.
  assert(DelayedInits.empty());
  assert(DelayedBBs.empty());
}

MDNode *llvm::MapMetadata(const MDNode *MD, ValueToValueMapTy &VM,
                          RemapFlags Flags, ValueMapTypeRemapper *TypeMapper,
                          ValueMaterializer *Materializer) {
  return cast_or_null<MDNode>(MapMetadata(static_cast<const Metadata *>(MD), VM,
                                          Flags, TypeMapper, Materializer));
}

void llvm::RemapInstruction(Instruction *I, ValueToValueMapTy &VM,
                            RemapFlags Flags, ValueMapTypeRemapper *TypeMapper,
                            ValueMaterializer *Materializer) {
  Mapper(VM, Flags, TypeMapper, Materializer).remapInstruction(I);
}

void Mapper::remapInstruction(Instruction *I) {
  // Remap operands.
  for (User::op_iterator op = I->op_begin(), E = I->op_end(); op != E; ++op) {
    Value *V = mapValue(*op);
    // If we aren't ignoring missing entries, assert that something happened.
    if (V)
      *op = V;
    else
      assert((Flags & RF_IgnoreMissingLocals) &&
             "Referenced value not in value map!");
  }

  // Remap phi nodes' incoming blocks.
  if (PHINode *PN = dyn_cast<PHINode>(I)) {
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
      Value *V = mapValue(PN->getIncomingBlock(i));
      // If we aren't ignoring missing entries, assert that something happened.
      if (V)
        PN->setIncomingBlock(i, cast<BasicBlock>(V));
      else
        assert((Flags & RF_IgnoreMissingLocals) &&
               "Referenced block not in value map!");
    }
  }

  // Remap attached metadata.
  SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
  I->getAllMetadata(MDs);
  for (const auto &MI : MDs) {
    MDNode *Old = MI.second;
    MDNode *New = cast_or_null<MDNode>(mapMetadata(Old));
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

void llvm::RemapFunction(Function &F, ValueToValueMapTy &VM, RemapFlags Flags,
                         ValueMapTypeRemapper *TypeMapper,
                         ValueMaterializer *Materializer) {
  Mapper(VM, Flags, TypeMapper, Materializer).remapFunction(F);
}

void Mapper::remapFunction(Function &F) {
  // Remap the operands.
  for (Use &Op : F.operands())
    if (Op)
      Op = mapValue(Op);

  // Remap the metadata attachments.
  SmallVector<std::pair<unsigned, MDNode *>, 8> MDs;
  F.getAllMetadata(MDs);
  for (const auto &I : MDs)
    F.setMetadata(I.first, cast_or_null<MDNode>(mapMetadata(I.second)));

  // Remap the argument types.
  if (TypeMapper)
    for (Argument &A : F.args())
      A.mutateType(TypeMapper->remapType(A.getType()));

  // Remap the instructions.
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      remapInstruction(&I);
}
