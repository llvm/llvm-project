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
#include "llvm/GlobalValue.h"
#include "llvm/Instruction.h"
using namespace llvm;

Value *llvm::MapValue(const Value *V, ValueMapTy &VM) {
  Value *&VMSlot = VM[V];
  if (VMSlot) return VMSlot;      // Does it exist in the map yet?
  
  // NOTE: VMSlot can be invalidated by any reference to VM, which can grow the
  // DenseMap.  This includes any recursive calls to MapValue.

  // Global values do not need to be seeded into the ValueMap if they are using
  // the identity mapping.
  if (isa<GlobalValue>(V) || isa<InlineAsm>(V))
    return VMSlot = const_cast<Value*>(V);

  if (Constant *C = const_cast<Constant*>(dyn_cast<Constant>(V))) {
    if (isa<ConstantInt>(C) || isa<ConstantFP>(C) ||
        isa<ConstantPointerNull>(C) || isa<ConstantAggregateZero>(C) ||
        isa<UndefValue>(C))
      return VMSlot = C;           // Primitive constants map directly
    else if (ConstantArray *CA = dyn_cast<ConstantArray>(C)) {
      for (User::op_iterator b = CA->op_begin(), i = b, e = CA->op_end();
           i != e; ++i) {
        Value *MV = MapValue(*i, VM);
        if (MV != *i) {
          // This array must contain a reference to a global, make a new array
          // and return it.
          //
          std::vector<Constant*> Values;
          Values.reserve(CA->getNumOperands());
          for (User::op_iterator j = b; j != i; ++j)
            Values.push_back(cast<Constant>(*j));
          Values.push_back(cast<Constant>(MV));
          for (++i; i != e; ++i)
            Values.push_back(cast<Constant>(MapValue(*i, VM)));
          return VM[V] = ConstantArray::get(CA->getType(), Values);
        }
      }
      return VM[V] = C;

    } else if (ConstantStruct *CS = dyn_cast<ConstantStruct>(C)) {
      for (User::op_iterator b = CS->op_begin(), i = b, e = CS->op_end();
           i != e; ++i) {
        Value *MV = MapValue(*i, VM);
        if (MV != *i) {
          // This struct must contain a reference to a global, make a new struct
          // and return it.
          //
          std::vector<Constant*> Values;
          Values.reserve(CS->getNumOperands());
          for (User::op_iterator j = b; j != i; ++j)
            Values.push_back(cast<Constant>(*j));
          Values.push_back(cast<Constant>(MV));
          for (++i; i != e; ++i)
            Values.push_back(cast<Constant>(MapValue(*i, VM)));
          return VM[V] = ConstantStruct::get(CS->getType(), Values);
        }
      }
      return VM[V] = C;

    } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
      std::vector<Constant*> Ops;
      for (User::op_iterator i = CE->op_begin(), e = CE->op_end(); i != e; ++i)
        Ops.push_back(cast<Constant>(MapValue(*i, VM)));
      return VM[V] = CE->getWithOperands(Ops);
    } else if (ConstantVector *CP = dyn_cast<ConstantVector>(C)) {
      for (User::op_iterator b = CP->op_begin(), i = b, e = CP->op_end();
           i != e; ++i) {
        Value *MV = MapValue(*i, VM);
        if (MV != *i) {
          // This vector value must contain a reference to a global, make a new
          // vector constant and return it.
          //
          std::vector<Constant*> Values;
          Values.reserve(CP->getNumOperands());
          for (User::op_iterator j = b; j != i; ++j)
            Values.push_back(cast<Constant>(*j));
          Values.push_back(cast<Constant>(MV));
          for (++i; i != e; ++i)
            Values.push_back(cast<Constant>(MapValue(*i, VM)));
          return VM[V] = ConstantVector::get(Values);
        }
      }
      return VM[V] = C;
      
    } else {
      assert(0 && "Unknown type of constant!");
    }
  }

  return 0;
}

/// RemapInstruction - Convert the instruction operands from referencing the
/// current values into those specified by ValueMap.
///
void llvm::RemapInstruction(Instruction *I, ValueMapTy &ValueMap) {
  for (User::op_iterator op = I->op_begin(), E = I->op_end(); op != E; ++op) {
    Value *V = MapValue(*op, ValueMap);
    assert(V && "Referenced value not in value map!");
    *op = V;
  }
}
