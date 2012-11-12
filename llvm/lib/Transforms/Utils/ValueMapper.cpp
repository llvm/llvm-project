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
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/Metadata.h"
using namespace llvm;

// Out of line method to get vtable etc for class.
void ValueMapTypeRemapper::anchor() {}

Value *llvm::MapValue(const Value *V, ValueToValueMapTy &VM, RemapFlags Flags,
                      ValueMapTypeRemapper *TypeMapper) {
  ValueToValueMapTy::iterator I = VM.find(V);
  
  // If the value already exists in the map, use it.
  if (I != VM.end() && I->second) return I->second;
  
  // Global values do not need to be seeded into the VM if they
  // are using the identity mapping.
  if (isa<GlobalValue>(V) || isa<MDString>(V))
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
  

  if (const MDNode *MD = dyn_cast<MDNode>(V)) {
    // If this is a module-level metadata and we know that nothing at the module
    // level is changing, then use an identity mapping.
    if (!MD->isFunctionLocal() && (Flags & RF_NoModuleLevelChanges))
      return VM[V] = const_cast<Value*>(V);
    
    // Create a dummy node in case we have a metadata cycle.
    MDNode *Dummy = MDNode::getTemporary(V->getContext(), ArrayRef<Value*>());
    VM[V] = Dummy;
    
    // Check all operands to see if any need to be remapped.
    for (unsigned i = 0, e = MD->getNumOperands(); i != e; ++i) {
      Value *OP = MD->getOperand(i);
      if (OP == 0 || MapValue(OP, VM, Flags, TypeMapper) == OP) continue;

      // Ok, at least one operand needs remapping.  
      SmallVector<Value*, 4> Elts;
      Elts.reserve(MD->getNumOperands());
      for (i = 0; i != e; ++i) {
        Value *Op = MD->getOperand(i);
        Elts.push_back(Op ? MapValue(Op, VM, Flags, TypeMapper) : 0);
      }
      MDNode *NewMD = MDNode::get(V->getContext(), Elts);
      Dummy->replaceAllUsesWith(NewMD);
      VM[V] = NewMD;
      MDNode::deleteTemporary(Dummy);
      return NewMD;
    }

    VM[V] = const_cast<Value*>(V);
    MDNode::deleteTemporary(Dummy);

    // No operands needed remapping.  Use an identity mapping.
    return const_cast<Value*>(V);
  }

  // Okay, this either must be a constant (which may or may not be mappable) or
  // is something that is not in the mapping table.
  Constant *C = const_cast<Constant*>(dyn_cast<Constant>(V));
  if (C == 0)
    return 0;
  
  if (BlockAddress *BA = dyn_cast<BlockAddress>(C)) {
    Function *F = 
      cast<Function>(MapValue(BA->getFunction(), VM, Flags, TypeMapper));
    BasicBlock *BB = cast_or_null<BasicBlock>(MapValue(BA->getBasicBlock(), VM,
                                                       Flags, TypeMapper));
    return VM[V] = BlockAddress::get(F, BB ? BB : BA->getBasicBlock());
  }
  
  // Otherwise, we have some other constant to remap.  Start by checking to see
  // if all operands have an identity remapping.
  unsigned OpNo = 0, NumOperands = C->getNumOperands();
  Value *Mapped = 0;
  for (; OpNo != NumOperands; ++OpNo) {
    Value *Op = C->getOperand(OpNo);
    Mapped = MapValue(Op, VM, Flags, TypeMapper);
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
                             Flags, TypeMapper));
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

/// RemapInstruction - Convert the instruction operands from referencing the
/// current values into those specified by VMap.
///
void llvm::RemapInstruction(Instruction *I, ValueToValueMapTy &VMap,
                            RemapFlags Flags, ValueMapTypeRemapper *TypeMapper){
  // Remap operands.
  for (User::op_iterator op = I->op_begin(), E = I->op_end(); op != E; ++op) {
    Value *V = MapValue(*op, VMap, Flags, TypeMapper);
    // If we aren't ignoring missing entries, assert that something happened.
    if (V != 0)
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
      if (V != 0)
        PN->setIncomingBlock(i, cast<BasicBlock>(V));
      else
        assert((Flags & RF_IgnoreMissingEntries) &&
               "Referenced block not in value map!");
    }
  }

  // Remap attached metadata.
  SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
  I->getAllMetadata(MDs);
  for (SmallVectorImpl<std::pair<unsigned, MDNode *> >::iterator
       MI = MDs.begin(), ME = MDs.end(); MI != ME; ++MI) {
    MDNode *Old = MI->second;
    MDNode *New = MapValue(Old, VMap, Flags, TypeMapper);
    if (New != Old)
      I->setMetadata(MI->first, New);
  }
  
  // If the instruction's type is being remapped, do so now.
  if (TypeMapper)
    I->mutateType(TypeMapper->remapType(I->getType()));
}
