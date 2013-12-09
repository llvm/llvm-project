//===-- AutoUpgrade.cpp - Implement auto-upgrade helper functions ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the auto-upgrade helper functions
//
//===----------------------------------------------------------------------===//

#include "llvm/AutoUpgrade.h"
#include "llvm/DebugInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstring>
using namespace llvm;

// Upgrade the declarations of the SSE4.1 functions whose arguments have
// changed their type from v4f32 to v2i64.
static bool UpgradeSSE41Function(Function* F, Intrinsic::ID IID,
                                 Function *&NewFn) {
  // Check whether this is an old version of the function, which received
  // v4f32 arguments.
  Type *Arg0Type = F->getFunctionType()->getParamType(0);
  if (Arg0Type != VectorType::get(Type::getFloatTy(F->getContext()), 4))
    return false;

  // Yes, it's old, replace it with new version.
  F->setName(F->getName() + ".old");
  NewFn = Intrinsic::getDeclaration(F->getParent(), IID);
  return true;
}

static bool UpgradeIntrinsicFunction1(Function *F, Function *&NewFn) {
  assert(F && "Illegal to upgrade a non-existent Function.");

  // Quickly eliminate it, if it's not a candidate.
  StringRef Name = F->getName();
  if (Name.size() <= 8 || !Name.startswith("llvm."))
    return false;
  Name = Name.substr(5); // Strip off "llvm."

  switch (Name[0]) {
  default: break;
  case 'a': {
    if (Name.startswith("arm.neon.vclz")) {
      Type* args[2] = {
        F->arg_begin()->getType(),
        Type::getInt1Ty(F->getContext())
      };
      // Can't use Intrinsic::getDeclaration here as it adds a ".i1" to
      // the end of the name. Change name from llvm.arm.neon.vclz.* to
      //  llvm.ctlz.*
      FunctionType* fType = FunctionType::get(F->getReturnType(), args, false);
      NewFn = Function::Create(fType, F->getLinkage(),
                               "llvm.ctlz." + Name.substr(14), F->getParent());
      return true;
    }
    if (Name.startswith("arm.neon.vcnt")) {
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::ctpop,
                                        F->arg_begin()->getType());
      return true;
    }
    break;
  }
  case 'c': {
    if (Name.startswith("ctlz.") && F->arg_size() == 1) {
      F->setName(Name + ".old");
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::ctlz,
                                        F->arg_begin()->getType());
      return true;
    }
    if (Name.startswith("cttz.") && F->arg_size() == 1) {
      F->setName(Name + ".old");
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::cttz,
                                        F->arg_begin()->getType());
      return true;
    }
    break;
  }
  case 'o':
    // We only need to change the name to match the mangling including the
    // address space.
    if (F->arg_size() == 2 && Name.startswith("objectsize.")) {
      Type *Tys[2] = { F->getReturnType(), F->arg_begin()->getType() };
      if (F->getName() != Intrinsic::getName(Intrinsic::objectsize, Tys)) {
        F->setName(Name + ".old");
        NewFn = Intrinsic::getDeclaration(F->getParent(),
                                          Intrinsic::objectsize, Tys);
        return true;
      }
    }
    break;

  case 'x': {
    if (Name.startswith("x86.sse2.pcmpeq.") ||
        Name.startswith("x86.sse2.pcmpgt.") ||
        Name.startswith("x86.avx2.pcmpeq.") ||
        Name.startswith("x86.avx2.pcmpgt.") ||
        Name.startswith("x86.avx.vpermil.") ||
        Name == "x86.avx.movnt.dq.256" ||
        Name == "x86.avx.movnt.pd.256" ||
        Name == "x86.avx.movnt.ps.256" ||
        Name == "x86.sse42.crc32.64.8" ||
        (Name.startswith("x86.xop.vpcom") && F->arg_size() == 2)) {
      NewFn = 0;
      return true;
    }
    // SSE4.1 ptest functions may have an old signature.
    if (Name.startswith("x86.sse41.ptest")) {
      if (Name == "x86.sse41.ptestc")
        return UpgradeSSE41Function(F, Intrinsic::x86_sse41_ptestc, NewFn);
      if (Name == "x86.sse41.ptestz")
        return UpgradeSSE41Function(F, Intrinsic::x86_sse41_ptestz, NewFn);
      if (Name == "x86.sse41.ptestnzc")
        return UpgradeSSE41Function(F, Intrinsic::x86_sse41_ptestnzc, NewFn);
    }
    // frcz.ss/sd may need to have an argument dropped
    if (Name.startswith("x86.xop.vfrcz.ss") && F->arg_size() == 2) {
      F->setName(Name + ".old");
      NewFn = Intrinsic::getDeclaration(F->getParent(),
                                        Intrinsic::x86_xop_vfrcz_ss);
      return true;
    }
    if (Name.startswith("x86.xop.vfrcz.sd") && F->arg_size() == 2) {
      F->setName(Name + ".old");
      NewFn = Intrinsic::getDeclaration(F->getParent(),
                                        Intrinsic::x86_xop_vfrcz_sd);
      return true;
    }
    // Fix the FMA4 intrinsics to remove the 4
    if (Name.startswith("x86.fma4.")) {
      F->setName("llvm.x86.fma" + Name.substr(8));
      NewFn = F;
      return true;
    }
    break;
  }
  }

  //  This may not belong here. This function is effectively being overloaded
  //  to both detect an intrinsic which needs upgrading, and to provide the
  //  upgraded form of the intrinsic. We should perhaps have two separate
  //  functions for this.
  return false;
}

bool llvm::UpgradeIntrinsicFunction(Function *F, Function *&NewFn) {
  NewFn = 0;
  bool Upgraded = UpgradeIntrinsicFunction1(F, NewFn);

  // Upgrade intrinsic attributes.  This does not change the function.
  if (NewFn)
    F = NewFn;
  if (unsigned id = F->getIntrinsicID())
    F->setAttributes(Intrinsic::getAttributes(F->getContext(),
                                              (Intrinsic::ID)id));
  return Upgraded;
}

bool llvm::UpgradeGlobalVariable(GlobalVariable *GV) {
  // Nothing to do yet.
  return false;
}

// UpgradeIntrinsicCall - Upgrade a call to an old intrinsic to be a call the
// upgraded intrinsic. All argument and return casting must be provided in
// order to seamlessly integrate with existing context.
void llvm::UpgradeIntrinsicCall(CallInst *CI, Function *NewFn) {
  Function *F = CI->getCalledFunction();
  LLVMContext &C = CI->getContext();
  IRBuilder<> Builder(C);
  Builder.SetInsertPoint(CI->getParent(), CI);

  assert(F && "Intrinsic call is not direct?");

  if (!NewFn) {
    // Get the Function's name.
    StringRef Name = F->getName();

    Value *Rep;
    // Upgrade packed integer vector compares intrinsics to compare instructions
    if (Name.startswith("llvm.x86.sse2.pcmpeq.") ||
        Name.startswith("llvm.x86.avx2.pcmpeq.")) {
      Rep = Builder.CreateICmpEQ(CI->getArgOperand(0), CI->getArgOperand(1),
                                 "pcmpeq");
      // need to sign extend since icmp returns vector of i1
      Rep = Builder.CreateSExt(Rep, CI->getType(), "");
    } else if (Name.startswith("llvm.x86.sse2.pcmpgt.") ||
               Name.startswith("llvm.x86.avx2.pcmpgt.")) {
      Rep = Builder.CreateICmpSGT(CI->getArgOperand(0), CI->getArgOperand(1),
                                  "pcmpgt");
      // need to sign extend since icmp returns vector of i1
      Rep = Builder.CreateSExt(Rep, CI->getType(), "");
    } else if (Name == "llvm.x86.avx.movnt.dq.256" ||
               Name == "llvm.x86.avx.movnt.ps.256" ||
               Name == "llvm.x86.avx.movnt.pd.256") {
      IRBuilder<> Builder(C);
      Builder.SetInsertPoint(CI->getParent(), CI);

      Module *M = F->getParent();
      SmallVector<Value *, 1> Elts;
      Elts.push_back(ConstantInt::get(Type::getInt32Ty(C), 1));
      MDNode *Node = MDNode::get(C, Elts);

      Value *Arg0 = CI->getArgOperand(0);
      Value *Arg1 = CI->getArgOperand(1);

      // Convert the type of the pointer to a pointer to the stored type.
      Value *BC = Builder.CreateBitCast(Arg0,
                                        PointerType::getUnqual(Arg1->getType()),
                                        "cast");
      StoreInst *SI = Builder.CreateStore(Arg1, BC);
      SI->setMetadata(M->getMDKindID("nontemporal"), Node);
      SI->setAlignment(16);

      // Remove intrinsic.
      CI->eraseFromParent();
      return;
    } else if (Name.startswith("llvm.x86.xop.vpcom")) {
      Intrinsic::ID intID;
      if (Name.endswith("ub"))
        intID = Intrinsic::x86_xop_vpcomub;
      else if (Name.endswith("uw"))
        intID = Intrinsic::x86_xop_vpcomuw;
      else if (Name.endswith("ud"))
        intID = Intrinsic::x86_xop_vpcomud;
      else if (Name.endswith("uq"))
        intID = Intrinsic::x86_xop_vpcomuq;
      else if (Name.endswith("b"))
        intID = Intrinsic::x86_xop_vpcomb;
      else if (Name.endswith("w"))
        intID = Intrinsic::x86_xop_vpcomw;
      else if (Name.endswith("d"))
        intID = Intrinsic::x86_xop_vpcomd;
      else if (Name.endswith("q"))
        intID = Intrinsic::x86_xop_vpcomq;
      else
        llvm_unreachable("Unknown suffix");

      Name = Name.substr(18); // strip off "llvm.x86.xop.vpcom"
      unsigned Imm;
      if (Name.startswith("lt"))
        Imm = 0;
      else if (Name.startswith("le"))
        Imm = 1;
      else if (Name.startswith("gt"))
        Imm = 2;
      else if (Name.startswith("ge"))
        Imm = 3;
      else if (Name.startswith("eq"))
        Imm = 4;
      else if (Name.startswith("ne"))
        Imm = 5;
      else if (Name.startswith("true"))
        Imm = 6;
      else if (Name.startswith("false"))
        Imm = 7;
      else
        llvm_unreachable("Unknown condition");

      Function *VPCOM = Intrinsic::getDeclaration(F->getParent(), intID);
      Rep = Builder.CreateCall3(VPCOM, CI->getArgOperand(0),
                                CI->getArgOperand(1), Builder.getInt8(Imm));
    } else if (Name == "llvm.x86.sse42.crc32.64.8") {
      Function *CRC32 = Intrinsic::getDeclaration(F->getParent(),
                                               Intrinsic::x86_sse42_crc32_32_8);
      Value *Trunc0 = Builder.CreateTrunc(CI->getArgOperand(0), Type::getInt32Ty(C));
      Rep = Builder.CreateCall2(CRC32, Trunc0, CI->getArgOperand(1));
      Rep = Builder.CreateZExt(Rep, CI->getType(), "");
    } else {
      bool PD128 = false, PD256 = false, PS128 = false, PS256 = false;
      if (Name == "llvm.x86.avx.vpermil.pd.256")
        PD256 = true;
      else if (Name == "llvm.x86.avx.vpermil.pd")
        PD128 = true;
      else if (Name == "llvm.x86.avx.vpermil.ps.256")
        PS256 = true;
      else if (Name == "llvm.x86.avx.vpermil.ps")
        PS128 = true;

      if (PD256 || PD128 || PS256 || PS128) {
        Value *Op0 = CI->getArgOperand(0);
        unsigned Imm = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
        SmallVector<Constant*, 8> Idxs;

        if (PD128)
          for (unsigned i = 0; i != 2; ++i)
            Idxs.push_back(Builder.getInt32((Imm >> i) & 0x1));
        else if (PD256)
          for (unsigned l = 0; l != 4; l+=2)
            for (unsigned i = 0; i != 2; ++i)
              Idxs.push_back(Builder.getInt32(((Imm >> (l+i)) & 0x1) + l));
        else if (PS128)
          for (unsigned i = 0; i != 4; ++i)
            Idxs.push_back(Builder.getInt32((Imm >> (2 * i)) & 0x3));
        else if (PS256)
          for (unsigned l = 0; l != 8; l+=4)
            for (unsigned i = 0; i != 4; ++i)
              Idxs.push_back(Builder.getInt32(((Imm >> (2 * i)) & 0x3) + l));
        else
          llvm_unreachable("Unexpected function");

        Rep = Builder.CreateShuffleVector(Op0, Op0, ConstantVector::get(Idxs));
      } else {
        llvm_unreachable("Unknown function for CallInst upgrade.");
      }
    }

    CI->replaceAllUsesWith(Rep);
    CI->eraseFromParent();
    return;
  }

  std::string Name = CI->getName().str();
  CI->setName(Name + ".old");

  switch (NewFn->getIntrinsicID()) {
  default:
    llvm_unreachable("Unknown function for CallInst upgrade.");

  case Intrinsic::ctlz:
  case Intrinsic::cttz:
    assert(CI->getNumArgOperands() == 1 &&
           "Mismatch between function args and call args");
    CI->replaceAllUsesWith(Builder.CreateCall2(NewFn, CI->getArgOperand(0),
                                               Builder.getFalse(), Name));
    CI->eraseFromParent();
    return;

  case Intrinsic::objectsize:
    CI->replaceAllUsesWith(Builder.CreateCall2(NewFn,
                                               CI->getArgOperand(0),
                                               CI->getArgOperand(1),
                                               Name));
    CI->eraseFromParent();
    return;

  case Intrinsic::arm_neon_vclz: {
    // Change name from llvm.arm.neon.vclz.* to llvm.ctlz.*
    CI->replaceAllUsesWith(Builder.CreateCall2(NewFn, CI->getArgOperand(0),
                                               Builder.getFalse(),
                                               "llvm.ctlz." + Name.substr(14)));
    CI->eraseFromParent();
    return;
  }
  case Intrinsic::ctpop: {
    CI->replaceAllUsesWith(Builder.CreateCall(NewFn, CI->getArgOperand(0)));
    CI->eraseFromParent();
    return;
  }

  case Intrinsic::x86_xop_vfrcz_ss:
  case Intrinsic::x86_xop_vfrcz_sd:
    CI->replaceAllUsesWith(Builder.CreateCall(NewFn, CI->getArgOperand(1),
                                              Name));
    CI->eraseFromParent();
    return;

  case Intrinsic::x86_sse41_ptestc:
  case Intrinsic::x86_sse41_ptestz:
  case Intrinsic::x86_sse41_ptestnzc: {
    // The arguments for these intrinsics used to be v4f32, and changed
    // to v2i64. This is purely a nop, since those are bitwise intrinsics.
    // So, the only thing required is a bitcast for both arguments.
    // First, check the arguments have the old type.
    Value *Arg0 = CI->getArgOperand(0);
    if (Arg0->getType() != VectorType::get(Type::getFloatTy(C), 4))
      return;

    // Old intrinsic, add bitcasts
    Value *Arg1 = CI->getArgOperand(1);

    Value *BC0 =
      Builder.CreateBitCast(Arg0,
                            VectorType::get(Type::getInt64Ty(C), 2),
                            "cast");
    Value *BC1 =
      Builder.CreateBitCast(Arg1,
                            VectorType::get(Type::getInt64Ty(C), 2),
                            "cast");

    CallInst* NewCall = Builder.CreateCall2(NewFn, BC0, BC1, Name);
    CI->replaceAllUsesWith(NewCall);
    CI->eraseFromParent();
    return;
  }
  }
}

// This tests each Function to determine if it needs upgrading. When we find
// one we are interested in, we then upgrade all calls to reflect the new
// function.
void llvm::UpgradeCallsToIntrinsic(Function* F) {
  assert(F && "Illegal attempt to upgrade a non-existent intrinsic.");

  // Upgrade the function and check if it is a totaly new function.
  Function *NewFn;
  if (UpgradeIntrinsicFunction(F, NewFn)) {
    if (NewFn != F) {
      // Replace all uses to the old function with the new one if necessary.
      for (Value::use_iterator UI = F->use_begin(), UE = F->use_end();
           UI != UE; ) {
        if (CallInst *CI = dyn_cast<CallInst>(*UI++))
          UpgradeIntrinsicCall(CI, NewFn);
      }
      // Remove old function, no longer used, from the module.
      F->eraseFromParent();
    }
  }
}

void llvm::UpgradeInstWithTBAATag(Instruction *I) {
  MDNode *MD = I->getMetadata(LLVMContext::MD_tbaa);
  assert(MD && "UpgradeInstWithTBAATag should have a TBAA tag");
  // Check if the tag uses struct-path aware TBAA format.
  if (isa<MDNode>(MD->getOperand(0)) && MD->getNumOperands() >= 3)
    return;

  if (MD->getNumOperands() == 3) {
    Value *Elts[] = {
      MD->getOperand(0),
      MD->getOperand(1)
    };
    MDNode *ScalarType = MDNode::get(I->getContext(), Elts);
    // Create a MDNode <ScalarType, ScalarType, offset 0, const>
    Value *Elts2[] = {
      ScalarType, ScalarType,
      Constant::getNullValue(Type::getInt64Ty(I->getContext())),
      MD->getOperand(2)
    };
    I->setMetadata(LLVMContext::MD_tbaa, MDNode::get(I->getContext(), Elts2));
  } else {
    // Create a MDNode <MD, MD, offset 0>
    Value *Elts[] = {MD, MD,
      Constant::getNullValue(Type::getInt64Ty(I->getContext()))};
    I->setMetadata(LLVMContext::MD_tbaa, MDNode::get(I->getContext(), Elts));
  }
}

Instruction *llvm::UpgradeBitCastInst(unsigned Opc, Value *V, Type *DestTy,
                                      Instruction *&Temp) {
  if (Opc != Instruction::BitCast)
    return 0;

  Temp = 0;
  Type *SrcTy = V->getType();
  if (SrcTy->isPtrOrPtrVectorTy() && DestTy->isPtrOrPtrVectorTy() &&
      SrcTy->getPointerAddressSpace() != DestTy->getPointerAddressSpace()) {
    LLVMContext &Context = V->getContext();

    // We have no information about target data layout, so we assume that
    // the maximum pointer size is 64bit.
    Type *MidTy = Type::getInt64Ty(Context);
    Temp = CastInst::Create(Instruction::PtrToInt, V, MidTy);

    return CastInst::Create(Instruction::IntToPtr, Temp, DestTy);
  }

  return 0;
}

Value *llvm::UpgradeBitCastExpr(unsigned Opc, Constant *C, Type *DestTy) {
  if (Opc != Instruction::BitCast)
    return 0;

  Type *SrcTy = C->getType();
  if (SrcTy->isPtrOrPtrVectorTy() && DestTy->isPtrOrPtrVectorTy() &&
      SrcTy->getPointerAddressSpace() != DestTy->getPointerAddressSpace()) {
    LLVMContext &Context = C->getContext();

    // We have no information about target data layout, so we assume that
    // the maximum pointer size is 64bit.
    Type *MidTy = Type::getInt64Ty(Context);

    return ConstantExpr::getIntToPtr(ConstantExpr::getPtrToInt(C, MidTy),
                                     DestTy);
  }

  return 0;
}

/// Check the debug info version number, if it is out-dated, drop the debug
/// info. Return true if module is modified.
bool llvm::UpgradeDebugInfo(Module &M) {
  if (getDebugMetadataVersionFromModule(M) == DEBUG_METADATA_VERSION)
    return false;

  return StripDebugInfo(M);
}
