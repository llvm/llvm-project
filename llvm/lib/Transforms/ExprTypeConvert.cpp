//===- ExprTypeConvert.cpp - Code to change an LLVM Expr Type -------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the part of level raising that checks to see if it is
// possible to coerce an entire expression tree into a different type.  If
// convertible, other routines from this file will do the conversion.
//
//===----------------------------------------------------------------------===//

#include "TransformInternals.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/Expressions.h"
#include "Support/STLExtras.h"
#include "Support/Debug.h"
#include <algorithm>
using namespace llvm;

static bool OperandConvertibleToType(User *U, Value *V, const Type *Ty,
                                     ValueTypeCache &ConvertedTypes,
                                     const TargetData &TD);

static void ConvertOperandToType(User *U, Value *OldVal, Value *NewVal,
                                 ValueMapCache &VMC, const TargetData &TD);

// Peephole Malloc instructions: we take a look at the use chain of the
// malloc instruction, and try to find out if the following conditions hold:
//   1. The malloc is of the form: 'malloc [sbyte], uint <constant>'
//   2. The only users of the malloc are cast & add instructions
//   3. Of the cast instructions, there is only one destination pointer type
//      [RTy] where the size of the pointed to object is equal to the number
//      of bytes allocated.
//
// If these conditions hold, we convert the malloc to allocate an [RTy]
// element.  TODO: This comment is out of date WRT arrays
//
static bool MallocConvertibleToType(MallocInst *MI, const Type *Ty,
                                    ValueTypeCache &CTMap,
                                    const TargetData &TD) {
  if (!isa<PointerType>(Ty)) return false;   // Malloc always returns pointers

  // Deal with the type to allocate, not the pointer type...
  Ty = cast<PointerType>(Ty)->getElementType();
  if (!Ty->isSized()) return false;      // Can only alloc something with a size

  // Analyze the number of bytes allocated...
  ExprType Expr = ClassifyExpr(MI->getArraySize());

  // Get information about the base datatype being allocated, before & after
  int ReqTypeSize = TD.getTypeSize(Ty);
  if (ReqTypeSize == 0) return false;
  unsigned OldTypeSize = TD.getTypeSize(MI->getType()->getElementType());

  // Must have a scale or offset to analyze it...
  if (!Expr.Offset && !Expr.Scale && OldTypeSize == 1) return false;

  // Get the offset and scale of the allocation...
  int64_t OffsetVal = Expr.Offset ? getConstantValue(Expr.Offset) : 0;
  int64_t ScaleVal = Expr.Scale ? getConstantValue(Expr.Scale) :(Expr.Var != 0);

  // The old type might not be of unit size, take old size into consideration
  // here...
  int64_t Offset = OffsetVal * OldTypeSize;
  int64_t Scale  = ScaleVal  * OldTypeSize;
  
  // In order to be successful, both the scale and the offset must be a multiple
  // of the requested data type's size.
  //
  if (Offset/ReqTypeSize*ReqTypeSize != Offset ||
      Scale/ReqTypeSize*ReqTypeSize != Scale)
    return false;   // Nope.

  return true;
}

static Instruction *ConvertMallocToType(MallocInst *MI, const Type *Ty,
                                        const std::string &Name,
                                        ValueMapCache &VMC,
                                        const TargetData &TD){
  BasicBlock *BB = MI->getParent();
  BasicBlock::iterator It = BB->end();

  // Analyze the number of bytes allocated...
  ExprType Expr = ClassifyExpr(MI->getArraySize());

  const PointerType *AllocTy = cast<PointerType>(Ty);
  const Type *ElType = AllocTy->getElementType();

  unsigned DataSize = TD.getTypeSize(ElType);
  unsigned OldTypeSize = TD.getTypeSize(MI->getType()->getElementType());

  // Get the offset and scale coefficients that we are allocating...
  int64_t OffsetVal = (Expr.Offset ? getConstantValue(Expr.Offset) : 0);
  int64_t ScaleVal = Expr.Scale ? getConstantValue(Expr.Scale) : (Expr.Var !=0);

  // The old type might not be of unit size, take old size into consideration
  // here...
  unsigned Offset = (uint64_t)OffsetVal * OldTypeSize / DataSize;
  unsigned Scale  = (uint64_t)ScaleVal  * OldTypeSize / DataSize;

  // Locate the malloc instruction, because we may be inserting instructions
  It = MI;

  // If we have a scale, apply it first...
  if (Expr.Var) {
    // Expr.Var is not necessarily unsigned right now, insert a cast now.
    if (Expr.Var->getType() != Type::UIntTy)
      Expr.Var = new CastInst(Expr.Var, Type::UIntTy,
                              Expr.Var->getName()+"-uint", It);

    if (Scale != 1)
      Expr.Var = BinaryOperator::create(Instruction::Mul, Expr.Var,
                                        ConstantUInt::get(Type::UIntTy, Scale),
                                        Expr.Var->getName()+"-scl", It);

  } else {
    // If we are not scaling anything, just make the offset be the "var"...
    Expr.Var = ConstantUInt::get(Type::UIntTy, Offset);
    Offset = 0; Scale = 1;
  }

  // If we have an offset now, add it in...
  if (Offset != 0) {
    assert(Expr.Var && "Var must be nonnull by now!");
    Expr.Var = BinaryOperator::create(Instruction::Add, Expr.Var,
                                      ConstantUInt::get(Type::UIntTy, Offset),
                                      Expr.Var->getName()+"-off", It);
  }

  assert(AllocTy == Ty);
  return new MallocInst(AllocTy->getElementType(), Expr.Var, Name);
}


// ExpressionConvertibleToType - Return true if it is possible
bool llvm::ExpressionConvertibleToType(Value *V, const Type *Ty,
                                 ValueTypeCache &CTMap, const TargetData &TD) {
  // Expression type must be holdable in a register.
  if (!Ty->isFirstClassType())
    return false;
  
  ValueTypeCache::iterator CTMI = CTMap.find(V);
  if (CTMI != CTMap.end()) return CTMI->second == Ty;

  // If it's a constant... all constants can be converted to a different
  // type.
  //
  if (isa<Constant>(V) && !isa<GlobalValue>(V))
    return true;
  
  CTMap[V] = Ty;
  if (V->getType() == Ty) return true;  // Expression already correct type!

  Instruction *I = dyn_cast<Instruction>(V);
  if (I == 0) return false;              // Otherwise, we can't convert!

  switch (I->getOpcode()) {
  case Instruction::Cast:
    // We can convert the expr if the cast destination type is losslessly
    // convertible to the requested type.
    if (!Ty->isLosslesslyConvertibleTo(I->getType())) return false;

    // We also do not allow conversion of a cast that casts from a ptr to array
    // of X to a *X.  For example: cast [4 x %List *] * %val to %List * *
    //
    if (const PointerType *SPT = 
        dyn_cast<PointerType>(I->getOperand(0)->getType()))
      if (const PointerType *DPT = dyn_cast<PointerType>(I->getType()))
        if (const ArrayType *AT = dyn_cast<ArrayType>(SPT->getElementType()))
          if (AT->getElementType() == DPT->getElementType())
            return false;
    break;

  case Instruction::Add:
  case Instruction::Sub:
    if (!Ty->isInteger() && !Ty->isFloatingPoint()) return false;
    if (!ExpressionConvertibleToType(I->getOperand(0), Ty, CTMap, TD) ||
        !ExpressionConvertibleToType(I->getOperand(1), Ty, CTMap, TD))
      return false;
    break;
  case Instruction::Shr:
    if (!Ty->isInteger()) return false;
    if (Ty->isSigned() != V->getType()->isSigned()) return false;
    // FALL THROUGH
  case Instruction::Shl:
    if (!Ty->isInteger()) return false;
    if (!ExpressionConvertibleToType(I->getOperand(0), Ty, CTMap, TD))
      return false;
    break;

  case Instruction::Load: {
    LoadInst *LI = cast<LoadInst>(I);
    if (!ExpressionConvertibleToType(LI->getPointerOperand(),
                                     PointerType::get(Ty), CTMap, TD))
      return false;
    break;                                     
  }
  case Instruction::PHI: {
    PHINode *PN = cast<PHINode>(I);
    // Be conservative if we find a giant PHI node.
    if (PN->getNumIncomingValues() > 32) return false;

    for (unsigned i = 0; i < PN->getNumIncomingValues(); ++i)
      if (!ExpressionConvertibleToType(PN->getIncomingValue(i), Ty, CTMap, TD))
        return false;
    break;
  }

  case Instruction::Malloc:
    if (!MallocConvertibleToType(cast<MallocInst>(I), Ty, CTMap, TD))
      return false;
    break;

  case Instruction::GetElementPtr: {
    // GetElementPtr's are directly convertible to a pointer type if they have
    // a number of zeros at the end.  Because removing these values does not
    // change the logical offset of the GEP, it is okay and fair to remove them.
    // This can change this:
    //   %t1 = getelementptr %Hosp * %hosp, ubyte 4, ubyte 0  ; <%List **>
    //   %t2 = cast %List * * %t1 to %List *
    // into
    //   %t2 = getelementptr %Hosp * %hosp, ubyte 4           ; <%List *>
    // 
    GetElementPtrInst *GEP = cast<GetElementPtrInst>(I);
    const PointerType *PTy = dyn_cast<PointerType>(Ty);
    if (!PTy) return false;  // GEP must always return a pointer...
    const Type *PVTy = PTy->getElementType();

    // Check to see if there are zero elements that we can remove from the
    // index array.  If there are, check to see if removing them causes us to
    // get to the right type...
    //
    std::vector<Value*> Indices(GEP->idx_begin(), GEP->idx_end());
    const Type *BaseType = GEP->getPointerOperand()->getType();
    const Type *ElTy = 0;

    while (!Indices.empty() &&
           Indices.back() == Constant::getNullValue(Indices.back()->getType())){
      Indices.pop_back();
      ElTy = GetElementPtrInst::getIndexedType(BaseType, Indices, true);
      if (ElTy == PVTy)
        break;  // Found a match!!
      ElTy = 0;
    }

    if (ElTy) break;   // Found a number of zeros we can strip off!

    // Otherwise, we can convert a GEP from one form to the other iff the
    // current gep is of the form 'getelementptr sbyte*, long N
    // and we could convert this to an appropriate GEP for the new type.
    //
    if (GEP->getNumOperands() == 2 &&
        GEP->getType() == PointerType::get(Type::SByteTy)) {

      // Do not Check to see if our incoming pointer can be converted
      // to be a ptr to an array of the right type... because in more cases than
      // not, it is simply not analyzable because of pointer/array
      // discrepancies.  To fix this, we will insert a cast before the GEP.
      //

      // Check to see if 'N' is an expression that can be converted to
      // the appropriate size... if so, allow it.
      //
      std::vector<Value*> Indices;
      const Type *ElTy = ConvertibleToGEP(PTy, I->getOperand(1), Indices, TD);
      if (ElTy == PVTy) {
        if (!ExpressionConvertibleToType(I->getOperand(0),
                                         PointerType::get(ElTy), CTMap, TD))
          return false;  // Can't continue, ExConToTy might have polluted set!
        break;
      }
    }

    // Otherwise, it could be that we have something like this:
    //     getelementptr [[sbyte] *] * %reg115, long %reg138    ; [sbyte]**
    // and want to convert it into something like this:
    //     getelemenptr [[int] *] * %reg115, long %reg138      ; [int]**
    //
    if (GEP->getNumOperands() == 2 && 
        PTy->getElementType()->isSized() &&
        TD.getTypeSize(PTy->getElementType()) == 
        TD.getTypeSize(GEP->getType()->getElementType())) {
      const PointerType *NewSrcTy = PointerType::get(PVTy);
      if (!ExpressionConvertibleToType(I->getOperand(0), NewSrcTy, CTMap, TD))
        return false;
      break;
    }

    return false;   // No match, maybe next time.
  }

  case Instruction::Call: {
    if (isa<Function>(I->getOperand(0)))
      return false;  // Don't even try to change direct calls.

    // If this is a function pointer, we can convert the return type if we can
    // convert the source function pointer.
    //
    const PointerType *PT = cast<PointerType>(I->getOperand(0)->getType());
    const FunctionType *FT = cast<FunctionType>(PT->getElementType());
    std::vector<const Type *> ArgTys(FT->param_begin(), FT->param_end());
    const FunctionType *NewTy =
      FunctionType::get(Ty, ArgTys, FT->isVarArg());
    if (!ExpressionConvertibleToType(I->getOperand(0),
                                     PointerType::get(NewTy), CTMap, TD))
      return false;
    break;
  }
  default:
    return false;
  }

  // Expressions are only convertible if all of the users of the expression can
  // have this value converted.  This makes use of the map to avoid infinite
  // recursion.
  //
  for (Value::use_iterator It = I->use_begin(), E = I->use_end(); It != E; ++It)
    if (!OperandConvertibleToType(*It, I, Ty, CTMap, TD))
      return false;

  return true;
}


Value *llvm::ConvertExpressionToType(Value *V, const Type *Ty, 
                                     ValueMapCache &VMC, const TargetData &TD) {
  if (V->getType() == Ty) return V;  // Already where we need to be?

  ValueMapCache::ExprMapTy::iterator VMCI = VMC.ExprMap.find(V);
  if (VMCI != VMC.ExprMap.end()) {
    const Value *GV = VMCI->second;
    const Type *GTy = VMCI->second->getType();
    assert(VMCI->second->getType() == Ty);

    if (Instruction *I = dyn_cast<Instruction>(V))
      ValueHandle IHandle(VMC, I);  // Remove I if it is unused now!

    return VMCI->second;
  }

  DEBUG(std::cerr << "CETT: " << (void*)V << " " << *V);

  Instruction *I = dyn_cast<Instruction>(V);
  if (I == 0) {
    Constant *CPV = cast<Constant>(V);
    // Constants are converted by constant folding the cast that is required.
    // We assume here that all casts are implemented for constant prop.
    Value *Result = ConstantExpr::getCast(CPV, Ty);
    // Add the instruction to the expression map
    //VMC.ExprMap[V] = Result;
    return Result;
  }


  BasicBlock *BB = I->getParent();
  std::string Name = I->getName();  if (!Name.empty()) I->setName("");
  Instruction *Res;     // Result of conversion

  ValueHandle IHandle(VMC, I);  // Prevent I from being removed!
  
  Constant *Dummy = Constant::getNullValue(Ty);

  switch (I->getOpcode()) {
  case Instruction::Cast:
    assert(VMC.NewCasts.count(ValueHandle(VMC, I)) == 0);
    Res = new CastInst(I->getOperand(0), Ty, Name);
    VMC.NewCasts.insert(ValueHandle(VMC, Res));
    break;
    
  case Instruction::Add:
  case Instruction::Sub:
    Res = BinaryOperator::create(cast<BinaryOperator>(I)->getOpcode(),
                                 Dummy, Dummy, Name);
    VMC.ExprMap[I] = Res;   // Add node to expression eagerly

    Res->setOperand(0, ConvertExpressionToType(I->getOperand(0), Ty, VMC, TD));
    Res->setOperand(1, ConvertExpressionToType(I->getOperand(1), Ty, VMC, TD));
    break;

  case Instruction::Shl:
  case Instruction::Shr:
    Res = new ShiftInst(cast<ShiftInst>(I)->getOpcode(), Dummy,
                        I->getOperand(1), Name);
    VMC.ExprMap[I] = Res;
    Res->setOperand(0, ConvertExpressionToType(I->getOperand(0), Ty, VMC, TD));
    break;

  case Instruction::Load: {
    LoadInst *LI = cast<LoadInst>(I);

    Res = new LoadInst(Constant::getNullValue(PointerType::get(Ty)), Name);
    VMC.ExprMap[I] = Res;
    Res->setOperand(0, ConvertExpressionToType(LI->getPointerOperand(),
                                               PointerType::get(Ty), VMC, TD));
    assert(Res->getOperand(0)->getType() == PointerType::get(Ty));
    assert(Ty == Res->getType());
    assert(Res->getType()->isFirstClassType() && "Load of structure or array!");
    break;
  }

  case Instruction::PHI: {
    PHINode *OldPN = cast<PHINode>(I);
    PHINode *NewPN = new PHINode(Ty, Name);

    VMC.ExprMap[I] = NewPN;   // Add node to expression eagerly
    while (OldPN->getNumOperands()) {
      BasicBlock *BB = OldPN->getIncomingBlock(0);
      Value *OldVal = OldPN->getIncomingValue(0);
      ValueHandle OldValHandle(VMC, OldVal);
      OldPN->removeIncomingValue(BB, false);
      Value *V = ConvertExpressionToType(OldVal, Ty, VMC, TD);
      NewPN->addIncoming(V, BB);
    }
    Res = NewPN;
    break;
  }

  case Instruction::Malloc: {
    Res = ConvertMallocToType(cast<MallocInst>(I), Ty, Name, VMC, TD);
    break;
  }

  case Instruction::GetElementPtr: {
    // GetElementPtr's are directly convertible to a pointer type if they have
    // a number of zeros at the end.  Because removing these values does not
    // change the logical offset of the GEP, it is okay and fair to remove them.
    // This can change this:
    //   %t1 = getelementptr %Hosp * %hosp, ubyte 4, ubyte 0  ; <%List **>
    //   %t2 = cast %List * * %t1 to %List *
    // into
    //   %t2 = getelementptr %Hosp * %hosp, ubyte 4           ; <%List *>
    // 
    GetElementPtrInst *GEP = cast<GetElementPtrInst>(I);

    // Check to see if there are zero elements that we can remove from the
    // index array.  If there are, check to see if removing them causes us to
    // get to the right type...
    //
    std::vector<Value*> Indices(GEP->idx_begin(), GEP->idx_end());
    const Type *BaseType = GEP->getPointerOperand()->getType();
    const Type *PVTy = cast<PointerType>(Ty)->getElementType();
    Res = 0;
    while (!Indices.empty() &&
           Indices.back() == Constant::getNullValue(Indices.back()->getType())){
      Indices.pop_back();
      if (GetElementPtrInst::getIndexedType(BaseType, Indices, true) == PVTy) {
        if (Indices.size() == 0)
          Res = new CastInst(GEP->getPointerOperand(), BaseType); // NOOP CAST
        else
          Res = new GetElementPtrInst(GEP->getPointerOperand(), Indices, Name);
        break;
      }
    }

    if (Res == 0 && GEP->getNumOperands() == 2 &&
        GEP->getType() == PointerType::get(Type::SByteTy)) {
      
      // Otherwise, we can convert a GEP from one form to the other iff the
      // current gep is of the form 'getelementptr sbyte*, unsigned N
      // and we could convert this to an appropriate GEP for the new type.
      //
      const PointerType *NewSrcTy = PointerType::get(PVTy);
      BasicBlock::iterator It = I;

      // Check to see if 'N' is an expression that can be converted to
      // the appropriate size... if so, allow it.
      //
      std::vector<Value*> Indices;
      const Type *ElTy = ConvertibleToGEP(NewSrcTy, I->getOperand(1),
                                          Indices, TD, &It);
      if (ElTy) {        
        assert(ElTy == PVTy && "Internal error, setup wrong!");
        Res = new GetElementPtrInst(Constant::getNullValue(NewSrcTy),
                                    Indices, Name);
        VMC.ExprMap[I] = Res;
        Res->setOperand(0, ConvertExpressionToType(I->getOperand(0),
                                                   NewSrcTy, VMC, TD));
      }
    }

    // Otherwise, it could be that we have something like this:
    //     getelementptr [[sbyte] *] * %reg115, uint %reg138    ; [sbyte]**
    // and want to convert it into something like this:
    //     getelemenptr [[int] *] * %reg115, uint %reg138      ; [int]**
    //
    if (Res == 0) {
      const PointerType *NewSrcTy = PointerType::get(PVTy);
      std::vector<Value*> Indices(GEP->idx_begin(), GEP->idx_end());
      Res = new GetElementPtrInst(Constant::getNullValue(NewSrcTy),
                                  Indices, Name);
      VMC.ExprMap[I] = Res;
      Res->setOperand(0, ConvertExpressionToType(I->getOperand(0),
                                                 NewSrcTy, VMC, TD));
    }


    assert(Res && "Didn't find match!");
    break;
  }

  case Instruction::Call: {
    assert(!isa<Function>(I->getOperand(0)));

    // If this is a function pointer, we can convert the return type if we can
    // convert the source function pointer.
    //
    const PointerType *PT = cast<PointerType>(I->getOperand(0)->getType());
    const FunctionType *FT = cast<FunctionType>(PT->getElementType());
    std::vector<const Type *> ArgTys(FT->param_begin(), FT->param_end());
    const FunctionType *NewTy =
      FunctionType::get(Ty, ArgTys, FT->isVarArg());
    const PointerType *NewPTy = PointerType::get(NewTy);
    if (Ty == Type::VoidTy)
      Name = "";  // Make sure not to name calls that now return void!

    Res = new CallInst(Constant::getNullValue(NewPTy),
                       std::vector<Value*>(I->op_begin()+1, I->op_end()),
                       Name);
    VMC.ExprMap[I] = Res;
    Res->setOperand(0, ConvertExpressionToType(I->getOperand(0),NewPTy,VMC,TD));
    break;
  }
  default:
    assert(0 && "Expression convertible, but don't know how to convert?");
    return 0;
  }

  assert(Res->getType() == Ty && "Didn't convert expr to correct type!");

  BB->getInstList().insert(I, Res);

  // Add the instruction to the expression map
  VMC.ExprMap[I] = Res;


  unsigned NumUses = I->use_size();
  for (unsigned It = 0; It < NumUses; ) {
    unsigned OldSize = NumUses;
    Value::use_iterator UI = I->use_begin();
    std::advance(UI, It);
    ConvertOperandToType(*UI, I, Res, VMC, TD);
    NumUses = I->use_size();
    if (NumUses == OldSize) ++It;
  }

  DEBUG(std::cerr << "ExpIn: " << (void*)I << " " << *I
                  << "ExpOut: " << (void*)Res << " " << *Res);

  return Res;
}



// ValueConvertibleToType - Return true if it is possible
bool llvm::ValueConvertibleToType(Value *V, const Type *Ty,
                                  ValueTypeCache &ConvertedTypes,
                                  const TargetData &TD) {
  ValueTypeCache::iterator I = ConvertedTypes.find(V);
  if (I != ConvertedTypes.end()) return I->second == Ty;
  ConvertedTypes[V] = Ty;

  // It is safe to convert the specified value to the specified type IFF all of
  // the uses of the value can be converted to accept the new typed value.
  //
  if (V->getType() != Ty) {
    for (Value::use_iterator I = V->use_begin(), E = V->use_end(); I != E; ++I)
      if (!OperandConvertibleToType(*I, V, Ty, ConvertedTypes, TD))
        return false;
  }

  return true;
}





// OperandConvertibleToType - Return true if it is possible to convert operand
// V of User (instruction) U to the specified type.  This is true iff it is
// possible to change the specified instruction to accept this.  CTMap is a map
// of converted types, so that circular definitions will see the future type of
// the expression, not the static current type.
//
static bool OperandConvertibleToType(User *U, Value *V, const Type *Ty,
                                     ValueTypeCache &CTMap,
                                     const TargetData &TD) {
  //  if (V->getType() == Ty) return true;   // Operand already the right type?

  // Expression type must be holdable in a register.
  if (!Ty->isFirstClassType())
    return false;

  Instruction *I = dyn_cast<Instruction>(U);
  if (I == 0) return false;              // We can't convert!

  switch (I->getOpcode()) {
  case Instruction::Cast:
    assert(I->getOperand(0) == V);
    // We can convert the expr if the cast destination type is losslessly
    // convertible to the requested type.
    // Also, do not change a cast that is a noop cast.  For all intents and
    // purposes it should be eliminated.
    if (!Ty->isLosslesslyConvertibleTo(I->getOperand(0)->getType()) ||
        I->getType() == I->getOperand(0)->getType())
      return false;

    // Do not allow a 'cast ushort %V to uint' to have it's first operand be
    // converted to a 'short' type.  Doing so changes the way sign promotion
    // happens, and breaks things.  Only allow the cast to take place if the
    // signedness doesn't change... or if the current cast is not a lossy
    // conversion.
    //
    if (!I->getType()->isLosslesslyConvertibleTo(I->getOperand(0)->getType()) &&
        I->getOperand(0)->getType()->isSigned() != Ty->isSigned())
      return false;

    // We also do not allow conversion of a cast that casts from a ptr to array
    // of X to a *X.  For example: cast [4 x %List *] * %val to %List * *
    //
    if (const PointerType *SPT = 
        dyn_cast<PointerType>(I->getOperand(0)->getType()))
      if (const PointerType *DPT = dyn_cast<PointerType>(I->getType()))
        if (const ArrayType *AT = dyn_cast<ArrayType>(SPT->getElementType()))
          if (AT->getElementType() == DPT->getElementType())
            return false;
    return true;

  case Instruction::Add:
    if (isa<PointerType>(Ty)) {
      Value *IndexVal = I->getOperand(V == I->getOperand(0) ? 1 : 0);
      std::vector<Value*> Indices;
      if (const Type *ETy = ConvertibleToGEP(Ty, IndexVal, Indices, TD)) {
        const Type *RetTy = PointerType::get(ETy);

        // Only successful if we can convert this type to the required type
        if (ValueConvertibleToType(I, RetTy, CTMap, TD)) {
          CTMap[I] = RetTy;
          return true;
        }
        // We have to return failure here because ValueConvertibleToType could 
        // have polluted our map
        return false;
      }
    }
    // FALLTHROUGH
  case Instruction::Sub: {
    if (!Ty->isInteger() && !Ty->isFloatingPoint()) return false;

    Value *OtherOp = I->getOperand((V == I->getOperand(0)) ? 1 : 0);
    return ValueConvertibleToType(I, Ty, CTMap, TD) &&
           ExpressionConvertibleToType(OtherOp, Ty, CTMap, TD);
  }
  case Instruction::SetEQ:
  case Instruction::SetNE: {
    Value *OtherOp = I->getOperand((V == I->getOperand(0)) ? 1 : 0);
    return ExpressionConvertibleToType(OtherOp, Ty, CTMap, TD);
  }
  case Instruction::Shr:
    if (Ty->isSigned() != V->getType()->isSigned()) return false;
    // FALL THROUGH
  case Instruction::Shl:
    if (I->getOperand(1) == V) return false;  // Cannot change shift amount type
    if (!Ty->isInteger()) return false;
    return ValueConvertibleToType(I, Ty, CTMap, TD);

  case Instruction::Free:
    assert(I->getOperand(0) == V);
    return isa<PointerType>(Ty);    // Free can free any pointer type!

  case Instruction::Load:
    // Cannot convert the types of any subscripts...
    if (I->getOperand(0) != V) return false;

    if (const PointerType *PT = dyn_cast<PointerType>(Ty)) {
      LoadInst *LI = cast<LoadInst>(I);
      
      const Type *LoadedTy = PT->getElementType();

      // They could be loading the first element of a composite type...
      if (const CompositeType *CT = dyn_cast<CompositeType>(LoadedTy)) {
        unsigned Offset = 0;     // No offset, get first leaf.
        std::vector<Value*> Indices;  // Discarded...
        LoadedTy = getStructOffsetType(CT, Offset, Indices, TD, false);
        assert(Offset == 0 && "Offset changed from zero???");
      }

      if (!LoadedTy->isFirstClassType())
        return false;

      if (TD.getTypeSize(LoadedTy) != TD.getTypeSize(LI->getType()))
        return false;

      return ValueConvertibleToType(LI, LoadedTy, CTMap, TD);
    }
    return false;

  case Instruction::Store: {
    StoreInst *SI = cast<StoreInst>(I);

    if (V == I->getOperand(0)) {
      ValueTypeCache::iterator CTMI = CTMap.find(I->getOperand(1));
      if (CTMI != CTMap.end()) {   // Operand #1 is in the table already?
        // If so, check to see if it's Ty*, or, more importantly, if it is a
        // pointer to a structure where the first element is a Ty... this code
        // is necessary because we might be trying to change the source and
        // destination type of the store (they might be related) and the dest
        // pointer type might be a pointer to structure.  Below we allow pointer
        // to structures where the 0th element is compatible with the value,
        // now we have to support the symmetrical part of this.
        //
        const Type *ElTy = cast<PointerType>(CTMI->second)->getElementType();

        // Already a pointer to what we want?  Trivially accept...
        if (ElTy == Ty) return true;

        // Tricky case now, if the destination is a pointer to structure,
        // obviously the source is not allowed to be a structure (cannot copy
        // a whole structure at a time), so the level raiser must be trying to
        // store into the first field.  Check for this and allow it now:
        //
        if (const StructType *SElTy = dyn_cast<StructType>(ElTy)) {
          unsigned Offset = 0;
          std::vector<Value*> Indices;
          ElTy = getStructOffsetType(ElTy, Offset, Indices, TD, false);
          assert(Offset == 0 && "Offset changed!");
          if (ElTy == 0)    // Element at offset zero in struct doesn't exist!
            return false;   // Can only happen for {}*
          
          if (ElTy == Ty)   // Looks like the 0th element of structure is
            return true;    // compatible!  Accept now!

          // Otherwise we know that we can't work, so just stop trying now.
          return false;
        }
      }

      // Can convert the store if we can convert the pointer operand to match
      // the new  value type...
      return ExpressionConvertibleToType(I->getOperand(1), PointerType::get(Ty),
                                         CTMap, TD);
    } else if (const PointerType *PT = dyn_cast<PointerType>(Ty)) {
      const Type *ElTy = PT->getElementType();
      assert(V == I->getOperand(1));

      if (isa<StructType>(ElTy)) {
        // We can change the destination pointer if we can store our first
        // argument into the first element of the structure...
        //
        unsigned Offset = 0;
        std::vector<Value*> Indices;
        ElTy = getStructOffsetType(ElTy, Offset, Indices, TD, false);
        assert(Offset == 0 && "Offset changed!");
        if (ElTy == 0)    // Element at offset zero in struct doesn't exist!
          return false;   // Can only happen for {}*
      }

      // Must move the same amount of data...
      if (!ElTy->isSized() || 
          TD.getTypeSize(ElTy) != TD.getTypeSize(I->getOperand(0)->getType()))
        return false;

      // Can convert store if the incoming value is convertible and if the
      // result will preserve semantics...
      const Type *Op0Ty = I->getOperand(0)->getType();
      if (!(Op0Ty->isIntegral() ^ ElTy->isIntegral()) &&
          !(Op0Ty->isFloatingPoint() ^ ElTy->isFloatingPoint()))
        return ExpressionConvertibleToType(I->getOperand(0), ElTy, CTMap, TD);
    }
    return false;
  }

  case Instruction::GetElementPtr:
    if (V != I->getOperand(0) || !isa<PointerType>(Ty)) return false;

    // If we have a two operand form of getelementptr, this is really little
    // more than a simple addition.  As with addition, check to see if the
    // getelementptr instruction can be changed to index into the new type.
    //
    if (I->getNumOperands() == 2) {
      const Type *OldElTy = cast<PointerType>(I->getType())->getElementType();
      unsigned DataSize = TD.getTypeSize(OldElTy);
      Value *Index = I->getOperand(1);
      Instruction *TempScale = 0;

      // If the old data element is not unit sized, we have to create a scale
      // instruction so that ConvertibleToGEP will know the REAL amount we are
      // indexing by.  Note that this is never inserted into the instruction
      // stream, so we have to delete it when we're done.
      //
      if (DataSize != 1) {
        Value *CST;
        if (Index->getType()->isSigned())
          CST = ConstantSInt::get(Index->getType(), DataSize);
        else
          CST = ConstantUInt::get(Index->getType(), DataSize);
                                  
        TempScale = BinaryOperator::create(Instruction::Mul, Index, CST);
        Index = TempScale;
      }

      // Check to see if the second argument is an expression that can
      // be converted to the appropriate size... if so, allow it.
      //
      std::vector<Value*> Indices;
      const Type *ElTy = ConvertibleToGEP(Ty, Index, Indices, TD);
      delete TempScale;   // Free our temporary multiply if we made it

      if (ElTy == 0) return false;  // Cannot make conversion...
      return ValueConvertibleToType(I, PointerType::get(ElTy), CTMap, TD);
    }
    return false;

  case Instruction::PHI: {
    PHINode *PN = cast<PHINode>(I);
    // Be conservative if we find a giant PHI node.
    if (PN->getNumIncomingValues() > 32) return false;

    for (unsigned i = 0; i < PN->getNumIncomingValues(); ++i)
      if (!ExpressionConvertibleToType(PN->getIncomingValue(i), Ty, CTMap, TD))
        return false;
    return ValueConvertibleToType(PN, Ty, CTMap, TD);
  }

  case Instruction::Call: {
    User::op_iterator OI = find(I->op_begin(), I->op_end(), V);
    assert (OI != I->op_end() && "Not using value!");
    unsigned OpNum = OI - I->op_begin();

    // Are we trying to change the function pointer value to a new type?
    if (OpNum == 0) {
      const PointerType *PTy = dyn_cast<PointerType>(Ty);
      if (PTy == 0) return false;  // Can't convert to a non-pointer type...
      const FunctionType *FTy = dyn_cast<FunctionType>(PTy->getElementType());
      if (FTy == 0) return false;  // Can't convert to a non ptr to function...

      // Do not allow converting to a call where all of the operands are ...'s
      if (FTy->getNumParams() == 0 && FTy->isVarArg())
        return false;              // Do not permit this conversion!

      // Perform sanity checks to make sure that new function type has the
      // correct number of arguments...
      //
      unsigned NumArgs = I->getNumOperands()-1;  // Don't include function ptr

      // Cannot convert to a type that requires more fixed arguments than
      // the call provides...
      //
      if (NumArgs < FTy->getNumParams()) return false;
      
      // Unless this is a vararg function type, we cannot provide more arguments
      // than are desired...
      //
      if (!FTy->isVarArg() && NumArgs > FTy->getNumParams())
        return false;

      // Okay, at this point, we know that the call and the function type match
      // number of arguments.  Now we see if we can convert the arguments
      // themselves.  Note that we do not require operands to be convertible,
      // we can insert casts if they are convertible but not compatible.  The
      // reason for this is that we prefer to have resolved functions but casted
      // arguments if possible.
      //
      for (unsigned i = 0, NA = FTy->getNumParams(); i < NA; ++i)
        if (!FTy->getParamType(i)->isLosslesslyConvertibleTo(I->getOperand(i+1)->getType()))
          return false;   // Operands must have compatible types!

      // Okay, at this point, we know that all of the arguments can be
      // converted.  We succeed if we can change the return type if
      // necessary...
      //
      return ValueConvertibleToType(I, FTy->getReturnType(), CTMap, TD);
    }
    
    const PointerType *MPtr = cast<PointerType>(I->getOperand(0)->getType());
    const FunctionType *FTy = cast<FunctionType>(MPtr->getElementType());
    if (!FTy->isVarArg()) return false;

    if ((OpNum-1) < FTy->getNumParams())
      return false;  // It's not in the varargs section...

    // If we get this far, we know the value is in the varargs section of the
    // function!  We can convert if we don't reinterpret the value...
    //
    return Ty->isLosslesslyConvertibleTo(V->getType());
  }
  }
  return false;
}


void llvm::ConvertValueToNewType(Value *V, Value *NewVal, ValueMapCache &VMC,
                                 const TargetData &TD) {
  ValueHandle VH(VMC, V);

  unsigned NumUses = V->use_size();
  for (unsigned It = 0; It < NumUses; ) {
    unsigned OldSize = NumUses;
    Value::use_iterator UI = V->use_begin();
    std::advance(UI, It);
    ConvertOperandToType(*UI, V, NewVal, VMC, TD);
    NumUses = V->use_size();
    if (NumUses == OldSize) ++It;
  }
}



static void ConvertOperandToType(User *U, Value *OldVal, Value *NewVal,
                                 ValueMapCache &VMC, const TargetData &TD) {
  if (isa<ValueHandle>(U)) return;  // Valuehandles don't let go of operands...

  if (VMC.OperandsMapped.count(U)) return;
  VMC.OperandsMapped.insert(U);

  ValueMapCache::ExprMapTy::iterator VMCI = VMC.ExprMap.find(U);
  if (VMCI != VMC.ExprMap.end())
    return;


  Instruction *I = cast<Instruction>(U);  // Only Instructions convertible

  BasicBlock *BB = I->getParent();
  assert(BB != 0 && "Instruction not embedded in basic block!");
  std::string Name = I->getName();
  I->setName("");
  Instruction *Res;     // Result of conversion

  //std::cerr << endl << endl << "Type:\t" << Ty << "\nInst: " << I
  //          << "BB Before: " << BB << endl;

  // Prevent I from being removed...
  ValueHandle IHandle(VMC, I);

  const Type *NewTy = NewVal->getType();
  Constant *Dummy = (NewTy != Type::VoidTy) ? 
                  Constant::getNullValue(NewTy) : 0;

  switch (I->getOpcode()) {
  case Instruction::Cast:
    if (VMC.NewCasts.count(ValueHandle(VMC, I))) {
      // This cast has already had it's value converted, causing a new cast to
      // be created.  We don't want to create YET ANOTHER cast instruction
      // representing the original one, so just modify the operand of this cast
      // instruction, which we know is newly created.
      I->setOperand(0, NewVal);
      I->setName(Name);  // give I its name back
      return;

    } else {
      Res = new CastInst(NewVal, I->getType(), Name);
    }
    break;

  case Instruction::Add:
    if (isa<PointerType>(NewTy)) {
      Value *IndexVal = I->getOperand(OldVal == I->getOperand(0) ? 1 : 0);
      std::vector<Value*> Indices;
      BasicBlock::iterator It = I;

      if (const Type *ETy = ConvertibleToGEP(NewTy, IndexVal, Indices, TD,&It)){
        // If successful, convert the add to a GEP
        //const Type *RetTy = PointerType::get(ETy);
        // First operand is actually the given pointer...
        Res = new GetElementPtrInst(NewVal, Indices, Name);
        assert(cast<PointerType>(Res->getType())->getElementType() == ETy &&
               "ConvertibleToGEP broken!");
        break;
      }
    }
    // FALLTHROUGH

  case Instruction::Sub:
  case Instruction::SetEQ:
  case Instruction::SetNE: {
    Res = BinaryOperator::create(cast<BinaryOperator>(I)->getOpcode(),
                                 Dummy, Dummy, Name);
    VMC.ExprMap[I] = Res;   // Add node to expression eagerly

    unsigned OtherIdx = (OldVal == I->getOperand(0)) ? 1 : 0;
    Value *OtherOp    = I->getOperand(OtherIdx);
    Res->setOperand(!OtherIdx, NewVal);
    Value *NewOther   = ConvertExpressionToType(OtherOp, NewTy, VMC, TD);
    Res->setOperand(OtherIdx, NewOther);
    break;
  }
  case Instruction::Shl:
  case Instruction::Shr:
    assert(I->getOperand(0) == OldVal);
    Res = new ShiftInst(cast<ShiftInst>(I)->getOpcode(), NewVal,
                        I->getOperand(1), Name);
    break;

  case Instruction::Free:            // Free can free any pointer type!
    assert(I->getOperand(0) == OldVal);
    Res = new FreeInst(NewVal);
    break;


  case Instruction::Load: {
    assert(I->getOperand(0) == OldVal && isa<PointerType>(NewVal->getType()));
    const Type *LoadedTy =
      cast<PointerType>(NewVal->getType())->getElementType();

    Value *Src = NewVal;

    if (const CompositeType *CT = dyn_cast<CompositeType>(LoadedTy)) {
      std::vector<Value*> Indices;
      Indices.push_back(Constant::getNullValue(Type::UIntTy));

      unsigned Offset = 0;   // No offset, get first leaf.
      LoadedTy = getStructOffsetType(CT, Offset, Indices, TD, false);
      assert(LoadedTy->isFirstClassType());

      if (Indices.size() != 1) {     // Do not generate load X, 0
        // Insert the GEP instruction before this load.
        Src = new GetElementPtrInst(Src, Indices, Name+".idx", I);
      }
    }
    
    Res = new LoadInst(Src, Name);
    assert(Res->getType()->isFirstClassType() && "Load of structure or array!");
    break;
  }

  case Instruction::Store: {
    if (I->getOperand(0) == OldVal) {  // Replace the source value
      // Check to see if operand #1 has already been converted...
      ValueMapCache::ExprMapTy::iterator VMCI =
        VMC.ExprMap.find(I->getOperand(1));
      if (VMCI != VMC.ExprMap.end()) {
        // Comments describing this stuff are in the OperandConvertibleToType
        // switch statement for Store...
        //
        const Type *ElTy =
          cast<PointerType>(VMCI->second->getType())->getElementType();
        
        Value *SrcPtr = VMCI->second;

        if (ElTy != NewTy) {
          // We check that this is a struct in the initial scan...
          const StructType *SElTy = cast<StructType>(ElTy);
          
          std::vector<Value*> Indices;
          Indices.push_back(Constant::getNullValue(Type::UIntTy));

          unsigned Offset = 0;
          const Type *Ty = getStructOffsetType(ElTy, Offset, Indices, TD,false);
          assert(Offset == 0 && "Offset changed!");
          assert(NewTy == Ty && "Did not convert to correct type!");

          // Insert the GEP instruction before this store.
          SrcPtr = new GetElementPtrInst(SrcPtr, Indices,
                                         SrcPtr->getName()+".idx", I);
        }
        Res = new StoreInst(NewVal, SrcPtr);

        VMC.ExprMap[I] = Res;
      } else {
        // Otherwise, we haven't converted Operand #1 over yet...
        const PointerType *NewPT = PointerType::get(NewTy);
        Res = new StoreInst(NewVal, Constant::getNullValue(NewPT));
        VMC.ExprMap[I] = Res;
        Res->setOperand(1, ConvertExpressionToType(I->getOperand(1),
                                                   NewPT, VMC, TD));
      }
    } else {                           // Replace the source pointer
      const Type *ValTy = cast<PointerType>(NewTy)->getElementType();

      Value *SrcPtr = NewVal;

      if (isa<StructType>(ValTy)) {
        std::vector<Value*> Indices;
        Indices.push_back(Constant::getNullValue(Type::UIntTy));

        unsigned Offset = 0;
        ValTy = getStructOffsetType(ValTy, Offset, Indices, TD, false);

        assert(Offset == 0 && ValTy);

        // Insert the GEP instruction before this store.
        SrcPtr = new GetElementPtrInst(SrcPtr, Indices,
                                       SrcPtr->getName()+".idx", I);
      }

      Res = new StoreInst(Constant::getNullValue(ValTy), SrcPtr);
      VMC.ExprMap[I] = Res;
      Res->setOperand(0, ConvertExpressionToType(I->getOperand(0),
                                                 ValTy, VMC, TD));
    }
    break;
  }


  case Instruction::GetElementPtr: {
    // Convert a one index getelementptr into just about anything that is
    // desired.
    //
    BasicBlock::iterator It = I;
    const Type *OldElTy = cast<PointerType>(I->getType())->getElementType();
    unsigned DataSize = TD.getTypeSize(OldElTy);
    Value *Index = I->getOperand(1);

    if (DataSize != 1) {
      // Insert a multiply of the old element type is not a unit size...
      Value *CST;
      if (Index->getType()->isSigned())
        CST = ConstantSInt::get(Index->getType(), DataSize);
      else
        CST = ConstantUInt::get(Index->getType(), DataSize);

      Index = BinaryOperator::create(Instruction::Mul, Index, CST, "scale", It);
    }

    // Perform the conversion now...
    //
    std::vector<Value*> Indices;
    const Type *ElTy = ConvertibleToGEP(NewVal->getType(),Index,Indices,TD,&It);
    assert(ElTy != 0 && "GEP Conversion Failure!");
    Res = new GetElementPtrInst(NewVal, Indices, Name);
    assert(Res->getType() == PointerType::get(ElTy) &&
           "ConvertibleToGet failed!");
  }
#if 0
    if (I->getType() == PointerType::get(Type::SByteTy)) {
      // Convert a getelementptr sbyte * %reg111, uint 16 freely back to
      // anything that is a pointer type...
      //
      BasicBlock::iterator It = I;
    
      // Check to see if the second argument is an expression that can
      // be converted to the appropriate size... if so, allow it.
      //
      std::vector<Value*> Indices;
      const Type *ElTy = ConvertibleToGEP(NewVal->getType(), I->getOperand(1),
                                          Indices, TD, &It);
      assert(ElTy != 0 && "GEP Conversion Failure!");
      
      Res = new GetElementPtrInst(NewVal, Indices, Name);
    } else {
      // Convert a getelementptr ulong * %reg123, uint %N
      // to        getelementptr  long * %reg123, uint %N
      // ... where the type must simply stay the same size...
      //
      GetElementPtrInst *GEP = cast<GetElementPtrInst>(I);
      std::vector<Value*> Indices(GEP->idx_begin(), GEP->idx_end());
      Res = new GetElementPtrInst(NewVal, Indices, Name);
    }
#endif
    break;

  case Instruction::PHI: {
    PHINode *OldPN = cast<PHINode>(I);
    PHINode *NewPN = new PHINode(NewTy, Name);
    VMC.ExprMap[I] = NewPN;

    while (OldPN->getNumOperands()) {
      BasicBlock *BB = OldPN->getIncomingBlock(0);
      Value *OldVal = OldPN->getIncomingValue(0);
      ValueHandle OldValHandle(VMC, OldVal);
      OldPN->removeIncomingValue(BB, false);
      Value *V = ConvertExpressionToType(OldVal, NewTy, VMC, TD);
      NewPN->addIncoming(V, BB);
    }
    Res = NewPN;
    break;
  }

  case Instruction::Call: {
    Value *Meth = I->getOperand(0);
    std::vector<Value*> Params(I->op_begin()+1, I->op_end());

    if (Meth == OldVal) {   // Changing the function pointer?
      const PointerType *NewPTy = cast<PointerType>(NewVal->getType());
      const FunctionType *NewTy = cast<FunctionType>(NewPTy->getElementType());

      if (NewTy->getReturnType() == Type::VoidTy)
        Name = "";  // Make sure not to name a void call!

      // Get an iterator to the call instruction so that we can insert casts for
      // operands if need be.  Note that we do not require operands to be
      // convertible, we can insert casts if they are convertible but not
      // compatible.  The reason for this is that we prefer to have resolved
      // functions but casted arguments if possible.
      //
      BasicBlock::iterator It = I;

      // Convert over all of the call operands to their new types... but only
      // convert over the part that is not in the vararg section of the call.
      //
      for (unsigned i = 0; i != NewTy->getNumParams(); ++i)
        if (Params[i]->getType() != NewTy->getParamType(i)) {
          // Create a cast to convert it to the right type, we know that this
          // is a lossless cast...
          //
          Params[i] = new CastInst(Params[i], NewTy->getParamType(i),
                                   "callarg.cast." +
                                   Params[i]->getName(), It);
        }
      Meth = NewVal;  // Update call destination to new value

    } else {                   // Changing an argument, must be in vararg area
      std::vector<Value*>::iterator OI =
        find(Params.begin(), Params.end(), OldVal);
      assert (OI != Params.end() && "Not using value!");

      *OI = NewVal;
    }

    Res = new CallInst(Meth, Params, Name);
    break;
  }
  default:
    assert(0 && "Expression convertible, but don't know how to convert?");
    return;
  }

  // If the instruction was newly created, insert it into the instruction
  // stream.
  //
  BasicBlock::iterator It = I;
  assert(It != BB->end() && "Instruction not in own basic block??");
  BB->getInstList().insert(It, Res);   // Keep It pointing to old instruction

  DEBUG(std::cerr << "COT CREATED: "  << (void*)Res << " " << *Res
                  << "In: " << (void*)I << " " << *I << "Out: " << (void*)Res
                  << " " << *Res);

  // Add the instruction to the expression map
  VMC.ExprMap[I] = Res;

  if (I->getType() != Res->getType())
    ConvertValueToNewType(I, Res, VMC, TD);
  else {
    bool FromStart = true;
    Value::use_iterator UI;
    while (1) {
      if (FromStart) UI = I->use_begin();
      if (UI == I->use_end()) break;
      
      if (isa<ValueHandle>(*UI)) {
        ++UI;
        FromStart = false;
      } else {
        User *U = *UI;
        if (!FromStart) --UI;
        U->replaceUsesOfWith(I, Res);
        if (!FromStart) ++UI;
      }
    }
  }
}


ValueHandle::ValueHandle(ValueMapCache &VMC, Value *V)
  : Instruction(Type::VoidTy, UserOp1, ""), Cache(VMC) {
  //DEBUG(std::cerr << "VH AQUIRING: " << (void*)V << " " << V);
  Operands.push_back(Use(V, this));
}

ValueHandle::ValueHandle(const ValueHandle &VH)
  : Instruction(Type::VoidTy, UserOp1, ""), Cache(VH.Cache) {
  //DEBUG(std::cerr << "VH AQUIRING: " << (void*)V << " " << V);
  Operands.push_back(Use((Value*)VH.getOperand(0), this));
}

static void RecursiveDelete(ValueMapCache &Cache, Instruction *I) {
  if (!I || !I->use_empty()) return;

  assert(I->getParent() && "Inst not in basic block!");

  //DEBUG(std::cerr << "VH DELETING: " << (void*)I << " " << I);

  for (User::op_iterator OI = I->op_begin(), OE = I->op_end(); 
       OI != OE; ++OI)
    if (Instruction *U = dyn_cast<Instruction>(OI)) {
      *OI = 0;
      RecursiveDelete(Cache, U);
    }

  I->getParent()->getInstList().remove(I);

  Cache.OperandsMapped.erase(I);
  Cache.ExprMap.erase(I);
  delete I;
}

ValueHandle::~ValueHandle() {
  if (Operands[0]->hasOneUse()) {
    Value *V = Operands[0];
    Operands[0] = 0;   // Drop use!

    // Now we just need to remove the old instruction so we don't get infinite
    // loops.  Note that we cannot use DCE because DCE won't remove a store
    // instruction, for example.
    //
    RecursiveDelete(Cache, dyn_cast<Instruction>(V));
  } else {
    //DEBUG(std::cerr << "VH RELEASING: " << (void*)Operands[0].get() << " "
    //                << Operands[0]->use_size() << " " << Operands[0]);
  }
}

