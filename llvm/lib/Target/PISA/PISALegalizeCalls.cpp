//===-- PISALegalizeCalls.cpp - modify function signatures ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Modify function argument/return types to the ones support by PISA:
// - natively supported types are used as is
//   - i8,i16,i32, and i64 scalar types
//   - vectors if 2,3, and 4 elements, with element types being i8,i16, and i64
//   - vectors if 2,3,4,5,6,7,8,16 and 32 elements, with element type being i32
// - vectors of a single element are scalarized
// - vectors of pointers are treated as equivalent native integer type
//   - vectors of p3/p4 pointers are treated as vectors of i32
//   - vectors of p0/p1/p2 pointers are treated as vectors of i64
// - i1 and i4 types are extended/truncated to i16
// - vectors of i1 are extended to ^2 size, cast to native integer type
// - all other types are passed in via memory argument
//===------------------------------------------------------------------===//

#include "PISA.h"
#include "llvm/IR/AttributeMask.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/PISAAddrSpace.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#define DEBUG_TYPE "pisa-legalize-calls"
#define DEBUG_NAME "PISA legalize calls"

using namespace llvm;

namespace {

class PISALegalizeCalls : public ModulePass,
                          public InstVisitor<PISALegalizeCalls> {
public:
  static char ID;
  PISALegalizeCalls() : ModulePass(ID) {}

  StringRef getPassName() const override { return DEBUG_NAME; }

  void visitCallInst(CallInst &CI);
  void visitIntrinsicInst(IntrinsicInst &I);
  void visitReturnInst(ReturnInst &RI);

private:
  SmallVector<Function *, 8> Funcs;
  SmallVector<CallInst *, 8> Calls;
  SmallVector<ReturnInst *, 8> Returns;

  bool runOnModule(Module &M) override;
  bool needsModification(Type *Ty, const DataLayout &DL);
  Type *getModifiedType(Type *, const DataLayout &, LLVMContext &);

  void collectFuncs(Function &F);
  void modifyFunctionSignature(Function &F);

  void modifyReturnInst(ReturnInst *RI);
  void modifyCallInst(CallInst *CI);
};
} // namespace

char PISALegalizeCalls::ID = 0;
INITIALIZE_PASS(PISALegalizeCalls, DEBUG_TYPE, DEBUG_NAME, false, false)

// determine if type will be modified in new function
bool PISALegalizeCalls::needsModification(Type *Ty, const DataLayout &DL) {
  bool Modify = false;
  if (Ty->isVoidTy() || Ty->isPointerTy())
    return false;

  if (FixedVectorType *VecTy = dyn_cast<FixedVectorType>(Ty)) {
    unsigned NumElts = VecTy->getNumElements();
    unsigned EltSize = VecTy->getScalarSizeInBits();
    if (EltSize == 0) {
      // vectors of pointers are treated as underlying integer
      assert(VecTy->getElementType()->isPointerTy());
      unsigned AS = VecTy->getElementType()->getPointerAddressSpace();
      EltSize = DL.getPointerSizeInBits(AS);
    }
    switch (EltSize) {
    case 0: { // vector of pointers
      assert(VecTy->getElementType()->isPointerTy());
      Modify = (NumElts > 4);
    } break;
    case 1:
    case 4:
      Modify = true;
      break;
    case 8:
    case 16:
    case 64:
      Modify = (NumElts == 1) || (NumElts > 4);
      break;
    case 32:
      Modify = (NumElts == 1) || ((NumElts > 8) && (NumElts != 16) &&
                                  (NumElts != 32) && (NumElts != 64));
      break;
    }
  } else if (IntegerType *IntTy = dyn_cast<IntegerType>(Ty)) {
    switch (IntTy->getScalarSizeInBits()) {
    default:
      reportFatalUsageError("unsupported PISA call integer type");
    case 1:
    case 4:
    case 128:
      Modify = true;
      break;
    case 8:
    case 16:
    case 32:
    case 64:
      break;
    }
  } else if (Ty->isFloatingPointTy()) {
    switch (Ty->getScalarSizeInBits()) {
    default:
      reportFatalUsageError("unsupported PISA call floating-point type");
    case 16:
    case 32:
    case 64:
      break;
    }
  } else if (isa<StructType>(Ty)) {
    Modify = true;
  } else {
    reportFatalUsageError("unsupported PISA call type");
  }
  return Modify;
}

Type *PISALegalizeCalls::getModifiedType(Type *Ty, const DataLayout &DL,
                                         LLVMContext &Ctx) {
  Type *NewTy = Ty;
  if (needsModification(Ty, DL)) {
    if (isa<IntegerType>(Ty)) {
      if (Ty->getScalarSizeInBits() == 1 || Ty->getScalarSizeInBits() == 4) {
        NewTy = IntegerType::get(Ctx, 16); // i{1, 4} => i16
      } else if (Ty->getScalarSizeInBits() == 128) {
        IntegerType *EltTy = IntegerType::get(Ctx, 64);
        NewTy = FixedVectorType::get(EltTy, 2); // i128 => v2i64
      } else {
        reportFatalUsageError("unsupported PISA call type conversion");
      }
    } else if (FixedVectorType *VecTy = dyn_cast<FixedVectorType>(Ty)) {
      if (VecTy->getNumElements() == 1) { // <1 x i?> => i?
        NewTy = VecTy->getElementType();
      } else if (VecTy->getScalarSizeInBits() == 1) { // <? x i1> => i16
        unsigned NewSize =
            std::max((unsigned)PowerOf2Ceil(VecTy->getNumElements()), 16u);
        NewTy = IntegerType::get(Ctx, NewSize);
      } else { // e.g. <8 x i8>
        int TotalSize = VecTy->getNumElements() * VecTy->getScalarSizeInBits();
        int NumEle = TotalSize / 32;
        if (NumEle > 0 && TotalSize % 32 == 0) {
          Type *NewElemTy = IntegerType::get(Ctx, 32);
          NewTy = FixedVectorType::get(NewElemTy, NumEle);
        }
        if (needsModification(NewTy, DL))
          NewTy =
              PointerType::get(Ctx, (unsigned)PISAAS::AddressSpace::PRIVATE);
      }
    } else {
      NewTy = PointerType::get(Ctx, (unsigned)PISAAS::AddressSpace::PRIVATE);
    }
  }
  return NewTy;
}

// record all call instructions that will require modification
void PISALegalizeCalls::visitCallInst(CallInst &CI) {
  Function *Caller = CI.getFunction();
  Function *Callee = CI.getCalledFunction();

  if (!Callee)
    Callee = dyn_cast<Function>(CI.getCalledOperand());
  if (!Callee && dyn_cast<InlineAsm>(CI.getCalledOperand()))
    return;

  assert((Callee || isa<PointerType>(CI.getCalledOperand()->getType())) &&
         "unable to extract callee info");

  // Indirect call
  if (!Callee && !isa<PointerType>(CI.getCalledOperand()->getType()))
    return;

  // call must with pisa cc
  if (CI.getCallingConv() == CallingConv::PISA_KERNEL)
    return;

  bool ArgNeedModification = llvm::any_of(CI.args(), [&](Value *Arg) {
    Type *SType = Arg->getType();
    return needsModification(SType, Caller->getParent()->getDataLayout());
  });
  if (ArgNeedModification ||
      needsModification(CI.getType(), Caller->getParent()->getDataLayout()))
    Calls.push_back(&CI);
}

void PISALegalizeCalls::visitIntrinsicInst(IntrinsicInst &I) {
  assert(!IntrinsicInst::mayLowerToFunctionCall(I.getIntrinsicID()) &&
         "Pre-Isel Intrinsics should be lowered to function calls by now");
  return;
}

// record all return instructions that will require modification
void PISALegalizeCalls::visitReturnInst(ReturnInst &RI) {
  Function *F = RI.getFunction();

  if (F->getCallingConv() == CallingConv::PISA_KERNEL)
    return;

  if (Value *RV = RI.getReturnValue()) {
    if (needsModification(RV->getType(), F->getParent()->getDataLayout())) {
      Returns.push_back(&RI);
    }
  }
}

void PISALegalizeCalls::collectFuncs(Function &F) {
  if (F.isIntrinsic() || F.getCallingConv() == CallingConv::PISA_KERNEL)
    return;

  bool ToAdd = false;
  for (Argument &Arg : F.args()) {
    if (needsModification(Arg.getType(), F.getParent()->getDataLayout()))
      ToAdd = true;
  }
  Type *Ty = F.getFunctionType()->getReturnType();
  if (needsModification(Ty, F.getParent()->getDataLayout()))
    ToAdd = true;

  if (ToAdd)
    Funcs.push_back(&F);
}

void PISALegalizeCalls::modifyFunctionSignature(Function &F) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  LLVMContext &Ctx = F.getContext();

  AttributeList AL = F.getAttributes();
  SmallVector<Type *> NewArgTys;
  for (unsigned I = 0, E = F.arg_size(); I < E; ++I) {
    Type *NewTy = getModifiedType(F.getArg(I)->getType(), DL, Ctx);
    NewArgTys.push_back(NewTy);
    AL = AL.removeParamAttributes(
        Ctx, I,
        AttributeFuncs::typeIncompatible(NewTy, F.getArg(I)->getAttributes()));
  }
  Type *RetTy = F.getFunctionType()->getReturnType();
  Type *NewRetTy = getModifiedType(RetTy, DL, Ctx);
  if (RetTy != NewRetTy) {
    // return via hidden memory arg
    if (NewRetTy->isPointerTy()) {
      NewArgTys.push_back(NewRetTy);
      NewRetTy = Type::getVoidTy(Ctx);
      // void functions cannot return any argument value
      for (const Argument &Arg : F.args())
        AL = AL.removeParamAttribute(Ctx, Arg.getArgNo(), Attribute::Returned);
    }
    AL = AL.removeRetAttributes(
        Ctx, AttributeFuncs::typeIncompatible(NewRetTy,
                                              F.getAttributes().getRetAttrs()));
  }

  // update function definition
  FunctionType *FTy = FunctionType::get(NewRetTy, NewArgTys, false);
  Function *NewF = Function::Create(FTy, F.getLinkage(), F.getAddressSpace());
  ValueToValueMapTy VMap;

  // map args 1:1, but types will be different.
  // modify actual references to args below.
  for (unsigned I = 0; I < F.arg_size(); I++) {
    Argument *SArg = F.getArg(I);
    VMap[SArg] = SArg;
  }

  SmallVector<ReturnInst *, 8> Returns;
  CloneFunctionInto(NewF, &F, VMap,
                    llvm::CloneFunctionChangeType::LocalChangesOnly, Returns,
                    "", 0);
  F.getParent()->getFunctionList().insert(F.getIterator(), NewF);

  // transform new arg types into ones expected within function
  IRBuilder<> IRB(Ctx);
  if (!NewF->isDeclaration()) {
    Function::iterator FirstBB = NewF->begin();
    IRB.SetInsertPoint(FirstBB->begin());
  }
  for (unsigned I = 0; I < F.arg_size(); I++) {
    Argument *SArg = F.getArg(I);
    Argument *DArg = NewF->getArg(I);
    Type *SType = SArg->getType(); // type to change from
    Type *DType = DArg->getType(); // type to change to
    if (!NewF->isDeclaration()) {
      if (SType != DType) {
        if (DType->isPointerTy()) { // value passed via memory
          LoadInst *Load = IRB.CreateLoad(SType, DArg);
          SArg->replaceAllUsesWith(Load);
        } else {
          TypeSize SSize = DL.getTypeSizeInBits(SType);
          TypeSize DSize = DL.getTypeSizeInBits(DType);
          if (SSize == DSize) { // <1 x i?> => i?
            Value *DCast = IRB.CreateBitCast(DArg, SType);
            SArg->replaceAllUsesWith(DCast);
          } else if (SType->isIntegerTy(1) ||
                     SType->isIntegerTy(4)) { // i8 => i{1, 4}
            Value *Trunc = IRB.CreateTrunc(DArg, SType);
            SArg->replaceAllUsesWith(Trunc);
          } else if (SType->isVectorTy()) { // i16 => <? x i1>
            assert(SType->getScalarSizeInBits() == 1);
            IntegerType *ScalarType = IntegerType::get(Ctx, SSize);
            Value *Trunc = IRB.CreateTrunc(DArg, ScalarType);
            Value *Cast = IRB.CreateBitCast(Trunc, SType);
            SArg->replaceAllUsesWith(Cast);
          } else {
            reportFatalUsageError("unsupported PISA function argument type");
          }
        }
      } else {
        SArg->replaceAllUsesWith(DArg);
      }
    }
  }
  // return modification handled in visitReturnInst
  NewF->takeName(&F);
  NewF->setAttributes(AL);
  NewF->setCallingConv(F.getCallingConv());
  F.replaceAllUsesWith(NewF);
  F.eraseFromParent();
}

void PISALegalizeCalls::modifyReturnInst(ReturnInst *RI) {
  Function *F = RI->getFunction();
  LLVMContext &Ctx = F->getContext();
  assert(F->getCallingConv() != CallingConv::PISA_KERNEL &&
         "return instruction in kernel");

  const DataLayout &DL = F->getParent()->getDataLayout();
  Value *RV = RI->getReturnValue();
  Type *SType = RV->getType();                         // type to change from
  Type *DType = F->getFunctionType()->getReturnType(); // type to change to
  if (!SType->isVoidTy() && (SType != DType)) {
    IRBuilder<> IRB(dyn_cast<Instruction>(RI));
    if (DType->isVoidTy()) {
      // return via hidden memory arg
      Argument *Ptr = F->getArg(F->arg_size() - 1);
      IRB.CreateStore(RV, Ptr);
      IRB.CreateRet(nullptr);
    } else {
      TypeSize SSize = DL.getTypeSizeInBits(SType);
      TypeSize DSize = DL.getTypeSizeInBits(DType);
      if (SSize == DSize) { // <1 x i?> => i?
        Value *DCast = IRB.CreateBitCast(RV, DType);
        IRB.CreateRet(DCast);
      } else if (SType->isIntegerTy(1) ||
                 SType->isIntegerTy(4)) { // i{1, 4} => i8
        bool SExt = F->hasRetAttribute(Attribute::SExt);
        Value *Extend =
            SExt ? IRB.CreateSExt(RV, DType) : IRB.CreateZExt(RV, DType);
        IRB.CreateRet(Extend);
      } else if (SType->isVectorTy()) { // <? x i1> => i16
        assert(SType->getScalarSizeInBits() == 1);
        IntegerType *ScalarType = IntegerType::get(Ctx, SSize);
        Value *Cast = IRB.CreateBitCast(RV, ScalarType);
        Value *Extend = IRB.CreateZExt(Cast, DType);
        IRB.CreateRet(Extend);
      } else {
        reportFatalUsageError("unsupported PISA function return type");
      }
    }
    RI->eraseFromParent();
  }
}

void PISALegalizeCalls::modifyCallInst(CallInst *CI) {
  Function *F = CI->getFunction();
  LLVMContext &Ctx = F->getContext();
  const DataLayout &DL = F->getParent()->getDataLayout();

  Value *Callee = CI->getCalledOperand();
  SmallVector<Value *, 8> NewArgs;
  SmallVector<Type *, 8> NewArgTys;

  IRBuilder<> IRB(dyn_cast<Instruction>(CI));
  for (unsigned I = 0; I < CI->arg_size(); I++) {
    Value *Arg = CI->getArgOperand(I);
    Type *SType = Arg->getType();                  // type to change from
    Type *DType = getModifiedType(SType, DL, Ctx); // type to change to
    Value *NewV = nullptr;
    if (SType != DType) {
      if (DType->isPointerTy()) { // pass via memory
        NewV = IRB.CreateAlloca(SType);
        IRB.CreateStore(Arg, NewV);
      } else {
        TypeSize SSize = DL.getTypeSizeInBits(SType);
        TypeSize DSize = DL.getTypeSizeInBits(DType);
        if (SSize == DSize) { // <1 x i?> => i?
          NewV = IRB.CreateBitCast(Arg, DType);
        } else if (SType->isIntegerTy(1) ||
                   SType->isIntegerTy(4)) { // i{1, 4} => i8
          bool SExt = CI->getParamAttr(I, Attribute::SExt).getKindAsEnum() ==
                      Attribute::SExt;
          NewV = SExt ? IRB.CreateSExt(Arg, DType) : IRB.CreateZExt(Arg, DType);
        } else if (SType->isVectorTy()) { // <? x i1> => i16
          assert(SType->getScalarSizeInBits() == 1);
          IntegerType *ScalarType = IntegerType::get(Ctx, SSize);
          Value *Cast = IRB.CreateBitCast(Arg, ScalarType);
          NewV = IRB.CreateZExt(Cast, DType);
        } else {
          reportFatalUsageError("unsupported PISA call argument type");
        }
      }
    }
    if (NewV) {
      NewArgs.push_back(NewV);
      NewArgTys.push_back(NewV->getType());
    } else {
      NewArgs.push_back(Arg);
      NewArgTys.push_back(Arg->getType());
    }
  }

  Type *RetTy = CI->getType();
  Type *NewRetTy = RetTy;
  Value *RetAlloca = nullptr;
  NewRetTy = getModifiedType(RetTy, DL, Ctx);
  if ((RetTy != NewRetTy) && NewRetTy->isPointerTy()) {
    // return via hidden memory arg
    RetAlloca = IRB.CreateAlloca(RetTy);
    NewArgs.push_back(RetAlloca);
    NewArgTys.push_back(NewRetTy);
    NewRetTy = Type::getVoidTy(Ctx);
  }
  FunctionType *FTy = FunctionType::get(NewRetTy, NewArgTys, false);
  CallInst *NewCI = IRB.CreateCall(FTy, Callee, NewArgs);
  NewCI->setCallingConv(CI->getCallingConv());

  // handle return value
  if (RetTy != NewRetTy) {
    TypeSize SSize = DL.getTypeSizeInBits(RetTy);
    TypeSize DSize = NewRetTy->isVoidTy()
                         ? TypeSize::getFixed(0)
                         : DL.getTypeSizeInBits(NewRetTy);
    if (SSize == DSize) { // i? => <1 x i?>
      Value *Bitcast = IRB.CreateBitCast(NewCI, RetTy);
      CI->replaceAllUsesWith(Bitcast);
    } else if (RetTy->isIntegerTy(1) ||
               RetTy->isIntegerTy(4)) { // i8 => i{1, 4}
      Value *Trunc = IRB.CreateTrunc(NewCI, RetTy);
      CI->replaceAllUsesWith(Trunc);
    } else if (DSize == 0) {
      // return via hidden memory arg
      LoadInst *Load = IRB.CreateLoad(RetTy, RetAlloca);
      CI->replaceAllUsesWith(Load);
    } else if (RetTy->isVectorTy()) { // i16 => <? x i1>
      assert(RetTy->getScalarSizeInBits() == 1);
      IntegerType *ScalarType = IntegerType::get(Ctx, SSize);
      Value *Trunc = IRB.CreateTrunc(NewCI, ScalarType);
      Value *Cast = IRB.CreateBitCast(Trunc, RetTy);
      CI->replaceAllUsesWith(Cast);
    } else {
      reportFatalUsageError("unsupported PISA call return type");
    }
  } else {
    CI->replaceAllUsesWith(NewCI);
  }
  CI->eraseFromParent();
}

bool PISALegalizeCalls::runOnModule(Module &M) {
  // record functions to be modified
  for (Function &F : M)
    collectFuncs(F);

  for (Function *F : Funcs)
    modifyFunctionSignature(*F);

  // record return/call instructions
  for (Function &F : M)
    visit(F);

  // modify return/call instructions
  for (ReturnInst *I : Returns)
    modifyReturnInst(I);
  for (CallInst *I : Calls)
    modifyCallInst(I);
  return !(Calls.empty() && Returns.empty() && Funcs.empty());
}

ModulePass *llvm::createPISALegalizeCallsPass() {
  return new PISALegalizeCalls();
}
