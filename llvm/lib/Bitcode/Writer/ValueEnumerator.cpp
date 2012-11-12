//===-- ValueEnumerator.cpp - Number values and types for bitcode writer --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ValueEnumerator class.
//
//===----------------------------------------------------------------------===//

#include "ValueEnumerator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/ValueSymbolTable.h"
#include "llvm/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace llvm;

static bool isIntegerValue(const std::pair<const Value*, unsigned> &V) {
  return V.first->getType()->isIntegerTy();
}

/// ValueEnumerator - Enumerate module-level information.
ValueEnumerator::ValueEnumerator(const Module *M) {
  // Enumerate the global variables.
  for (Module::const_global_iterator I = M->global_begin(),
         E = M->global_end(); I != E; ++I)
    EnumerateValue(I);

  // Enumerate the functions.
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I) {
    EnumerateValue(I);
    EnumerateAttributes(cast<Function>(I)->getAttributes());
  }

  // Enumerate the aliases.
  for (Module::const_alias_iterator I = M->alias_begin(), E = M->alias_end();
       I != E; ++I)
    EnumerateValue(I);

  // Remember what is the cutoff between globalvalue's and other constants.
  unsigned FirstConstant = Values.size();

  // Enumerate the global variable initializers.
  for (Module::const_global_iterator I = M->global_begin(),
         E = M->global_end(); I != E; ++I)
    if (I->hasInitializer())
      EnumerateValue(I->getInitializer());

  // Enumerate the aliasees.
  for (Module::const_alias_iterator I = M->alias_begin(), E = M->alias_end();
       I != E; ++I)
    EnumerateValue(I->getAliasee());

  // Insert constants and metadata that are named at module level into the slot 
  // pool so that the module symbol table can refer to them...
  EnumerateValueSymbolTable(M->getValueSymbolTable());
  EnumerateNamedMetadata(M);

  SmallVector<std::pair<unsigned, MDNode*>, 8> MDs;

  // Enumerate types used by function bodies and argument lists.
  for (Module::const_iterator F = M->begin(), E = M->end(); F != E; ++F) {

    for (Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
         I != E; ++I)
      EnumerateType(I->getType());

    for (Function::const_iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
      for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E;++I){
        for (User::const_op_iterator OI = I->op_begin(), E = I->op_end();
             OI != E; ++OI) {
          if (MDNode *MD = dyn_cast<MDNode>(*OI))
            if (MD->isFunctionLocal() && MD->getFunction())
              // These will get enumerated during function-incorporation.
              continue;
          EnumerateOperandType(*OI);
        }
        EnumerateType(I->getType());
        if (const CallInst *CI = dyn_cast<CallInst>(I))
          EnumerateAttributes(CI->getAttributes());
        else if (const InvokeInst *II = dyn_cast<InvokeInst>(I))
          EnumerateAttributes(II->getAttributes());

        // Enumerate metadata attached with this instruction.
        MDs.clear();
        I->getAllMetadataOtherThanDebugLoc(MDs);
        for (unsigned i = 0, e = MDs.size(); i != e; ++i)
          EnumerateMetadata(MDs[i].second);
        
        if (!I->getDebugLoc().isUnknown()) {
          MDNode *Scope, *IA;
          I->getDebugLoc().getScopeAndInlinedAt(Scope, IA, I->getContext());
          if (Scope) EnumerateMetadata(Scope);
          if (IA) EnumerateMetadata(IA);
        }
      }
  }

  // Optimize constant ordering.
  OptimizeConstants(FirstConstant, Values.size());
}

unsigned ValueEnumerator::getInstructionID(const Instruction *Inst) const {
  InstructionMapType::const_iterator I = InstructionMap.find(Inst);
  assert(I != InstructionMap.end() && "Instruction is not mapped!");
  return I->second;
}

void ValueEnumerator::setInstructionID(const Instruction *I) {
  InstructionMap[I] = InstructionCount++;
}

unsigned ValueEnumerator::getValueID(const Value *V) const {
  if (isa<MDNode>(V) || isa<MDString>(V)) {
    ValueMapType::const_iterator I = MDValueMap.find(V);
    assert(I != MDValueMap.end() && "Value not in slotcalculator!");
    return I->second-1;
  }

  ValueMapType::const_iterator I = ValueMap.find(V);
  assert(I != ValueMap.end() && "Value not in slotcalculator!");
  return I->second-1;
}

void ValueEnumerator::dump() const {
  print(dbgs(), ValueMap, "Default");
  dbgs() << '\n';
  print(dbgs(), MDValueMap, "MetaData");
  dbgs() << '\n';
}

void ValueEnumerator::print(raw_ostream &OS, const ValueMapType &Map,
                            const char *Name) const {

  OS << "Map Name: " << Name << "\n";
  OS << "Size: " << Map.size() << "\n";
  for (ValueMapType::const_iterator I = Map.begin(),
         E = Map.end(); I != E; ++I) {

    const Value *V = I->first;
    if (V->hasName())
      OS << "Value: " << V->getName();
    else
      OS << "Value: [null]\n";
    V->dump();

    OS << " Uses(" << std::distance(V->use_begin(),V->use_end()) << "):";
    for (Value::const_use_iterator UI = V->use_begin(), UE = V->use_end();
         UI != UE; ++UI) {
      if (UI != V->use_begin())
        OS << ",";
      if((*UI)->hasName())
        OS << " " << (*UI)->getName();
      else
        OS << " [null]";

    }
    OS <<  "\n\n";
  }
}

// Optimize constant ordering.
namespace {
  struct CstSortPredicate {
    ValueEnumerator &VE;
    explicit CstSortPredicate(ValueEnumerator &ve) : VE(ve) {}
    bool operator()(const std::pair<const Value*, unsigned> &LHS,
                    const std::pair<const Value*, unsigned> &RHS) {
      // Sort by plane.
      if (LHS.first->getType() != RHS.first->getType())
        return VE.getTypeID(LHS.first->getType()) <
               VE.getTypeID(RHS.first->getType());
      // Then by frequency.
      return LHS.second > RHS.second;
    }
  };
}

/// OptimizeConstants - Reorder constant pool for denser encoding.
void ValueEnumerator::OptimizeConstants(unsigned CstStart, unsigned CstEnd) {
  if (CstStart == CstEnd || CstStart+1 == CstEnd) return;

  CstSortPredicate P(*this);
  std::stable_sort(Values.begin()+CstStart, Values.begin()+CstEnd, P);

  // Ensure that integer constants are at the start of the constant pool.  This
  // is important so that GEP structure indices come before gep constant exprs.
  std::partition(Values.begin()+CstStart, Values.begin()+CstEnd,
                 isIntegerValue);

  // Rebuild the modified portion of ValueMap.
  for (; CstStart != CstEnd; ++CstStart)
    ValueMap[Values[CstStart].first] = CstStart+1;
}


/// EnumerateValueSymbolTable - Insert all of the values in the specified symbol
/// table into the values table.
void ValueEnumerator::EnumerateValueSymbolTable(const ValueSymbolTable &VST) {
  for (ValueSymbolTable::const_iterator VI = VST.begin(), VE = VST.end();
       VI != VE; ++VI)
    EnumerateValue(VI->getValue());
}

/// EnumerateNamedMetadata - Insert all of the values referenced by
/// named metadata in the specified module.
void ValueEnumerator::EnumerateNamedMetadata(const Module *M) {
  for (Module::const_named_metadata_iterator I = M->named_metadata_begin(),
       E = M->named_metadata_end(); I != E; ++I)
    EnumerateNamedMDNode(I);
}

void ValueEnumerator::EnumerateNamedMDNode(const NamedMDNode *MD) {
  for (unsigned i = 0, e = MD->getNumOperands(); i != e; ++i)
    EnumerateMetadata(MD->getOperand(i));
}

/// EnumerateMDNodeOperands - Enumerate all non-function-local values
/// and types referenced by the given MDNode.
void ValueEnumerator::EnumerateMDNodeOperands(const MDNode *N) {
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    if (Value *V = N->getOperand(i)) {
      if (isa<MDNode>(V) || isa<MDString>(V))
        EnumerateMetadata(V);
      else if (!isa<Instruction>(V) && !isa<Argument>(V))
        EnumerateValue(V);
    } else
      EnumerateType(Type::getVoidTy(N->getContext()));
  }
}

void ValueEnumerator::EnumerateMetadata(const Value *MD) {
  assert((isa<MDNode>(MD) || isa<MDString>(MD)) && "Invalid metadata kind");

  // Enumerate the type of this value.
  EnumerateType(MD->getType());

  const MDNode *N = dyn_cast<MDNode>(MD);

  // In the module-level pass, skip function-local nodes themselves, but
  // do walk their operands.
  if (N && N->isFunctionLocal() && N->getFunction()) {
    EnumerateMDNodeOperands(N);
    return;
  }

  // Check to see if it's already in!
  unsigned &MDValueID = MDValueMap[MD];
  if (MDValueID) {
    // Increment use count.
    MDValues[MDValueID-1].second++;
    return;
  }
  MDValues.push_back(std::make_pair(MD, 1U));
  MDValueID = MDValues.size();

  // Enumerate all non-function-local operands.
  if (N)
    EnumerateMDNodeOperands(N);
}

/// EnumerateFunctionLocalMetadataa - Incorporate function-local metadata
/// information reachable from the given MDNode.
void ValueEnumerator::EnumerateFunctionLocalMetadata(const MDNode *N) {
  assert(N->isFunctionLocal() && N->getFunction() &&
         "EnumerateFunctionLocalMetadata called on non-function-local mdnode!");

  // Enumerate the type of this value.
  EnumerateType(N->getType());

  // Check to see if it's already in!
  unsigned &MDValueID = MDValueMap[N];
  if (MDValueID) {
    // Increment use count.
    MDValues[MDValueID-1].second++;
    return;
  }
  MDValues.push_back(std::make_pair(N, 1U));
  MDValueID = MDValues.size();

  // To incoroporate function-local information visit all function-local
  // MDNodes and all function-local values they reference.
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
    if (Value *V = N->getOperand(i)) {
      if (MDNode *O = dyn_cast<MDNode>(V)) {
        if (O->isFunctionLocal() && O->getFunction())
          EnumerateFunctionLocalMetadata(O);
      } else if (isa<Instruction>(V) || isa<Argument>(V))
        EnumerateValue(V);
    }

  // Also, collect all function-local MDNodes for easy access.
  FunctionLocalMDs.push_back(N);
}

void ValueEnumerator::EnumerateValue(const Value *V) {
  assert(!V->getType()->isVoidTy() && "Can't insert void values!");
  assert(!isa<MDNode>(V) && !isa<MDString>(V) &&
         "EnumerateValue doesn't handle Metadata!");

  // Check to see if it's already in!
  unsigned &ValueID = ValueMap[V];
  if (ValueID) {
    // Increment use count.
    Values[ValueID-1].second++;
    return;
  }

  // Enumerate the type of this value.
  EnumerateType(V->getType());

  if (const Constant *C = dyn_cast<Constant>(V)) {
    if (isa<GlobalValue>(C)) {
      // Initializers for globals are handled explicitly elsewhere.
    } else if (C->getNumOperands()) {
      // If a constant has operands, enumerate them.  This makes sure that if a
      // constant has uses (for example an array of const ints), that they are
      // inserted also.

      // We prefer to enumerate them with values before we enumerate the user
      // itself.  This makes it more likely that we can avoid forward references
      // in the reader.  We know that there can be no cycles in the constants
      // graph that don't go through a global variable.
      for (User::const_op_iterator I = C->op_begin(), E = C->op_end();
           I != E; ++I)
        if (!isa<BasicBlock>(*I)) // Don't enumerate BB operand to BlockAddress.
          EnumerateValue(*I);

      // Finally, add the value.  Doing this could make the ValueID reference be
      // dangling, don't reuse it.
      Values.push_back(std::make_pair(V, 1U));
      ValueMap[V] = Values.size();
      return;
    }
  }

  // Add the value.
  Values.push_back(std::make_pair(V, 1U));
  ValueID = Values.size();
}


void ValueEnumerator::EnumerateType(Type *Ty) {
  unsigned *TypeID = &TypeMap[Ty];

  // We've already seen this type.
  if (*TypeID)
    return;

  // If it is a non-anonymous struct, mark the type as being visited so that we
  // don't recursively visit it.  This is safe because we allow forward
  // references of these in the bitcode reader.
  if (StructType *STy = dyn_cast<StructType>(Ty))
    if (!STy->isLiteral())
      *TypeID = ~0U;
  
  // Enumerate all of the subtypes before we enumerate this type.  This ensures
  // that the type will be enumerated in an order that can be directly built.
  for (Type::subtype_iterator I = Ty->subtype_begin(), E = Ty->subtype_end();
       I != E; ++I)
    EnumerateType(*I);
  
  // Refresh the TypeID pointer in case the table rehashed.
  TypeID = &TypeMap[Ty];
  
  // Check to see if we got the pointer another way.  This can happen when
  // enumerating recursive types that hit the base case deeper than they start.
  //
  // If this is actually a struct that we are treating as forward ref'able,
  // then emit the definition now that all of its contents are available.
  if (*TypeID && *TypeID != ~0U)
    return;
  
  // Add this type now that its contents are all happily enumerated.
  Types.push_back(Ty);
  
  *TypeID = Types.size();
}

// Enumerate the types for the specified value.  If the value is a constant,
// walk through it, enumerating the types of the constant.
void ValueEnumerator::EnumerateOperandType(const Value *V) {
  EnumerateType(V->getType());
  
  if (const Constant *C = dyn_cast<Constant>(V)) {
    // If this constant is already enumerated, ignore it, we know its type must
    // be enumerated.
    if (ValueMap.count(V)) return;

    // This constant may have operands, make sure to enumerate the types in
    // them.
    for (unsigned i = 0, e = C->getNumOperands(); i != e; ++i) {
      const Value *Op = C->getOperand(i);
      
      // Don't enumerate basic blocks here, this happens as operands to
      // blockaddress.
      if (isa<BasicBlock>(Op)) continue;
      
      EnumerateOperandType(Op);
    }

    if (const MDNode *N = dyn_cast<MDNode>(V)) {
      for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
        if (Value *Elem = N->getOperand(i))
          EnumerateOperandType(Elem);
    }
  } else if (isa<MDString>(V) || isa<MDNode>(V))
    EnumerateMetadata(V);
}

void ValueEnumerator::EnumerateAttributes(const AttrListPtr &PAL) {
  if (PAL.isEmpty()) return;  // null is always 0.
  // Do a lookup.
  unsigned &Entry = AttributeMap[PAL.getRawPointer()];
  if (Entry == 0) {
    // Never saw this before, add it.
    Attributes.push_back(PAL);
    Entry = Attributes.size();
  }
}

void ValueEnumerator::incorporateFunction(const Function &F) {
  InstructionCount = 0;
  NumModuleValues = Values.size();
  NumModuleMDValues = MDValues.size();

  // Adding function arguments to the value table.
  for (Function::const_arg_iterator I = F.arg_begin(), E = F.arg_end();
       I != E; ++I)
    EnumerateValue(I);

  FirstFuncConstantID = Values.size();

  // Add all function-level constants to the value table.
  for (Function::const_iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E; ++I)
      for (User::const_op_iterator OI = I->op_begin(), E = I->op_end();
           OI != E; ++OI) {
        if ((isa<Constant>(*OI) && !isa<GlobalValue>(*OI)) ||
            isa<InlineAsm>(*OI))
          EnumerateValue(*OI);
      }
    BasicBlocks.push_back(BB);
    ValueMap[BB] = BasicBlocks.size();
  }

  // Optimize the constant layout.
  OptimizeConstants(FirstFuncConstantID, Values.size());

  // Add the function's parameter attributes so they are available for use in
  // the function's instruction.
  EnumerateAttributes(F.getAttributes());

  FirstInstID = Values.size();

  SmallVector<MDNode *, 8> FnLocalMDVector;
  // Add all of the instructions.
  for (Function::const_iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E; ++I) {
      for (User::const_op_iterator OI = I->op_begin(), E = I->op_end();
           OI != E; ++OI) {
        if (MDNode *MD = dyn_cast<MDNode>(*OI))
          if (MD->isFunctionLocal() && MD->getFunction())
            // Enumerate metadata after the instructions they might refer to.
            FnLocalMDVector.push_back(MD);
      }

      SmallVector<std::pair<unsigned, MDNode*>, 8> MDs;
      I->getAllMetadataOtherThanDebugLoc(MDs);
      for (unsigned i = 0, e = MDs.size(); i != e; ++i) {
        MDNode *N = MDs[i].second;
        if (N->isFunctionLocal() && N->getFunction())
          FnLocalMDVector.push_back(N);
      }
        
      if (!I->getType()->isVoidTy())
        EnumerateValue(I);
    }
  }

  // Add all of the function-local metadata.
  for (unsigned i = 0, e = FnLocalMDVector.size(); i != e; ++i)
    EnumerateFunctionLocalMetadata(FnLocalMDVector[i]);
}

void ValueEnumerator::purgeFunction() {
  /// Remove purged values from the ValueMap.
  for (unsigned i = NumModuleValues, e = Values.size(); i != e; ++i)
    ValueMap.erase(Values[i].first);
  for (unsigned i = NumModuleMDValues, e = MDValues.size(); i != e; ++i)
    MDValueMap.erase(MDValues[i].first);
  for (unsigned i = 0, e = BasicBlocks.size(); i != e; ++i)
    ValueMap.erase(BasicBlocks[i]);

  Values.resize(NumModuleValues);
  MDValues.resize(NumModuleMDValues);
  BasicBlocks.clear();
  FunctionLocalMDs.clear();
}

static void IncorporateFunctionInfoGlobalBBIDs(const Function *F,
                                 DenseMap<const BasicBlock*, unsigned> &IDMap) {
  unsigned Counter = 0;
  for (Function::const_iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
    IDMap[BB] = ++Counter;
}

/// getGlobalBasicBlockID - This returns the function-specific ID for the
/// specified basic block.  This is relatively expensive information, so it
/// should only be used by rare constructs such as address-of-label.
unsigned ValueEnumerator::getGlobalBasicBlockID(const BasicBlock *BB) const {
  unsigned &Idx = GlobalBasicBlockIDs[BB];
  if (Idx != 0)
    return Idx-1;

  IncorporateFunctionInfoGlobalBBIDs(BB->getParent(), GlobalBasicBlockIDs);
  return getGlobalBasicBlockID(BB);
}

