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
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/IRBuilder.h"
#include <cstring>
using namespace llvm;


static bool UpgradeIntrinsicFunction1(Function *F, Function *&NewFn) {
  assert(F && "Illegal to upgrade a non-existent Function.");

  // Get the Function's name.
  const std::string& Name = F->getName();

  // Convenience
  const FunctionType *FTy = F->getFunctionType();

  // Quickly eliminate it, if it's not a candidate.
  if (Name.length() <= 8 || Name[0] != 'l' || Name[1] != 'l' || 
      Name[2] != 'v' || Name[3] != 'm' || Name[4] != '.')
    return false;

  Module *M = F->getParent();
  switch (Name[5]) {
  default: break;
  case 'a':
    // This upgrades the llvm.atomic.lcs, llvm.atomic.las, llvm.atomic.lss,
    // and atomics with default address spaces to their new names to their new
    // function name (e.g. llvm.atomic.add.i32 => llvm.atomic.add.i32.p0i32)
    if (Name.compare(5,7,"atomic.",7) == 0) {
      if (Name.compare(12,3,"lcs",3) == 0) {
        std::string::size_type delim = Name.find('.',12);
        F->setName("llvm.atomic.cmp.swap" + Name.substr(delim) +
                   ".p0" + Name.substr(delim+1));
        NewFn = F;
        return true;
      }
      else if (Name.compare(12,3,"las",3) == 0) {
        std::string::size_type delim = Name.find('.',12);
        F->setName("llvm.atomic.load.add"+Name.substr(delim)
                   + ".p0" + Name.substr(delim+1));
        NewFn = F;
        return true;
      }
      else if (Name.compare(12,3,"lss",3) == 0) {
        std::string::size_type delim = Name.find('.',12);
        F->setName("llvm.atomic.load.sub"+Name.substr(delim)
                   + ".p0" + Name.substr(delim+1));
        NewFn = F;
        return true;
      }
      else if (Name.rfind(".p") == std::string::npos) {
        // We don't have an address space qualifier so this has be upgraded
        // to the new name.  Copy the type name at the end of the intrinsic
        // and add to it
        std::string::size_type delim = Name.find_last_of('.');
        assert(delim != std::string::npos && "can not find type");
        F->setName(Name + ".p0" + Name.substr(delim+1));
        NewFn = F;
        return true;
      }
    } else if (Name.compare(5, 9, "arm.neon.", 9) == 0) {
      if (((Name.compare(14, 5, "vmovl", 5) == 0 ||
            Name.compare(14, 5, "vaddl", 5) == 0 ||
            Name.compare(14, 5, "vsubl", 5) == 0 ||
            Name.compare(14, 5, "vaddw", 5) == 0 ||
            Name.compare(14, 5, "vsubw", 5) == 0 ||
            Name.compare(14, 5, "vmull", 5) == 0 ||
            Name.compare(14, 5, "vmlal", 5) == 0 ||
            Name.compare(14, 5, "vmlsl", 5) == 0 ||
            Name.compare(14, 5, "vabdl", 5) == 0 ||
            Name.compare(14, 5, "vabal", 5) == 0) &&
           (Name.compare(19, 2, "s.", 2) == 0 ||
            Name.compare(19, 2, "u.", 2) == 0)) ||

          (Name.compare(14, 4, "vaba", 4) == 0 &&
           (Name.compare(18, 2, "s.", 2) == 0 ||
            Name.compare(18, 2, "u.", 2) == 0)) ||

          (Name.compare(14, 6, "vmovn.", 6) == 0)) {

        // Calls to these are transformed into IR without intrinsics.
        NewFn = 0;
        return true;
      }
      // Old versions of NEON ld/st intrinsics are missing alignment arguments.
      bool isVLd = (Name.compare(14, 3, "vld", 3) == 0);
      bool isVSt = (Name.compare(14, 3, "vst", 3) == 0);
      if (isVLd || isVSt) {
        unsigned NumVecs = Name.at(17) - '0';
        if (NumVecs == 0 || NumVecs > 4)
          return false;
        bool isLaneOp = (Name.compare(18, 5, "lane.", 5) == 0);
        if (!isLaneOp && Name.at(18) != '.')
          return false;
        unsigned ExpectedArgs = 2; // for the address and alignment
        if (isVSt || isLaneOp)
          ExpectedArgs += NumVecs;
        if (isLaneOp)
          ExpectedArgs += 1; // for the lane number
        unsigned NumP = FTy->getNumParams();
        if (NumP != ExpectedArgs - 1)
          return false;

        // Change the name of the old (bad) intrinsic, because 
        // its type is incorrect, but we cannot overload that name.
        F->setName("");

        // One argument is missing: add the alignment argument.
        std::vector<const Type*> NewParams;
        for (unsigned p = 0; p < NumP; ++p)
          NewParams.push_back(FTy->getParamType(p));
        NewParams.push_back(Type::getInt32Ty(F->getContext()));
        FunctionType *NewFTy = FunctionType::get(FTy->getReturnType(),
                                                 NewParams, false);
        NewFn = cast<Function>(M->getOrInsertFunction(Name, NewFTy));
        return true;
      }
    }
    break;
  case 'b':
    //  This upgrades the name of the llvm.bswap intrinsic function to only use 
    //  a single type name for overloading. We only care about the old format
    //  'llvm.bswap.i*.i*', so check for 'bswap.' and then for there being 
    //  a '.' after 'bswap.'
    if (Name.compare(5,6,"bswap.",6) == 0) {
      std::string::size_type delim = Name.find('.',11);
      
      if (delim != std::string::npos) {
        //  Construct the new name as 'llvm.bswap' + '.i*'
        F->setName(Name.substr(0,10)+Name.substr(delim));
        NewFn = F;
        return true;
      }
    }
    break;

  case 'c':
    //  We only want to fix the 'llvm.ct*' intrinsics which do not have the 
    //  correct return type, so we check for the name, and then check if the 
    //  return type does not match the parameter type.
    if ( (Name.compare(5,5,"ctpop",5) == 0 ||
          Name.compare(5,4,"ctlz",4) == 0 ||
          Name.compare(5,4,"cttz",4) == 0) &&
        FTy->getReturnType() != FTy->getParamType(0)) {
      //  We first need to change the name of the old (bad) intrinsic, because 
      //  its type is incorrect, but we cannot overload that name. We 
      //  arbitrarily unique it here allowing us to construct a correctly named 
      //  and typed function below.
      F->setName("");

      //  Now construct the new intrinsic with the correct name and type. We 
      //  leave the old function around in order to query its type, whatever it 
      //  may be, and correctly convert up to the new type.
      NewFn = cast<Function>(M->getOrInsertFunction(Name, 
                                                    FTy->getParamType(0),
                                                    FTy->getParamType(0),
                                                    (Type *)0));
      return true;
    }
    break;

  case 'e':
    //  The old llvm.eh.selector.i32 is equivalent to the new llvm.eh.selector.
    if (Name.compare("llvm.eh.selector.i32") == 0) {
      F->setName("llvm.eh.selector");
      NewFn = F;
      return true;
    }
    //  The old llvm.eh.typeid.for.i32 is equivalent to llvm.eh.typeid.for.
    if (Name.compare("llvm.eh.typeid.for.i32") == 0) {
      F->setName("llvm.eh.typeid.for");
      NewFn = F;
      return true;
    }
    //  Convert the old llvm.eh.selector.i64 to a call to llvm.eh.selector.
    if (Name.compare("llvm.eh.selector.i64") == 0) {
      NewFn = Intrinsic::getDeclaration(M, Intrinsic::eh_selector);
      return true;
    }
    //  Convert the old llvm.eh.typeid.for.i64 to a call to llvm.eh.typeid.for.
    if (Name.compare("llvm.eh.typeid.for.i64") == 0) {
      NewFn = Intrinsic::getDeclaration(M, Intrinsic::eh_typeid_for);
      return true;
    }
    break;

  case 'm': {
    // This upgrades the llvm.memcpy, llvm.memmove, and llvm.memset to the
    // new format that allows overloading the pointer for different address
    // space (e.g., llvm.memcpy.i16 => llvm.memcpy.p0i8.p0i8.i16)
    const char* NewFnName = NULL;
    if (Name.compare(5,8,"memcpy.i",8) == 0) {
      if (Name[13] == '8')
        NewFnName = "llvm.memcpy.p0i8.p0i8.i8";
      else if (Name.compare(13,2,"16") == 0)
        NewFnName = "llvm.memcpy.p0i8.p0i8.i16";
      else if (Name.compare(13,2,"32") == 0)
        NewFnName = "llvm.memcpy.p0i8.p0i8.i32";
      else if (Name.compare(13,2,"64") == 0)
        NewFnName = "llvm.memcpy.p0i8.p0i8.i64";
    } else if (Name.compare(5,9,"memmove.i",9) == 0) {
      if (Name[14] == '8')
        NewFnName = "llvm.memmove.p0i8.p0i8.i8";
      else if (Name.compare(14,2,"16") == 0)
        NewFnName = "llvm.memmove.p0i8.p0i8.i16";
      else if (Name.compare(14,2,"32") == 0)
        NewFnName = "llvm.memmove.p0i8.p0i8.i32";
      else if (Name.compare(14,2,"64") == 0)
        NewFnName = "llvm.memmove.p0i8.p0i8.i64";
    }
    else if (Name.compare(5,8,"memset.i",8) == 0) {
      if (Name[13] == '8')
        NewFnName = "llvm.memset.p0i8.i8";
      else if (Name.compare(13,2,"16") == 0)
        NewFnName = "llvm.memset.p0i8.i16";
      else if (Name.compare(13,2,"32") == 0)
        NewFnName = "llvm.memset.p0i8.i32";
      else if (Name.compare(13,2,"64") == 0)
        NewFnName = "llvm.memset.p0i8.i64";
    }
    if (NewFnName) {
      NewFn = cast<Function>(M->getOrInsertFunction(NewFnName, 
                                            FTy->getReturnType(),
                                            FTy->getParamType(0),
                                            FTy->getParamType(1),
                                            FTy->getParamType(2),
                                            FTy->getParamType(3),
                                            Type::getInt1Ty(F->getContext()),
                                            (Type *)0));
      return true;
    }
    break;
  }
  case 'p':
    //  This upgrades the llvm.part.select overloaded intrinsic names to only 
    //  use one type specifier in the name. We only care about the old format
    //  'llvm.part.select.i*.i*', and solve as above with bswap.
    if (Name.compare(5,12,"part.select.",12) == 0) {
      std::string::size_type delim = Name.find('.',17);
      
      if (delim != std::string::npos) {
        //  Construct a new name as 'llvm.part.select' + '.i*'
        F->setName(Name.substr(0,16)+Name.substr(delim));
        NewFn = F;
        return true;
      }
      break;
    }

    //  This upgrades the llvm.part.set intrinsics similarly as above, however 
    //  we care about 'llvm.part.set.i*.i*.i*', but only the first two types 
    //  must match. There is an additional type specifier after these two 
    //  matching types that we must retain when upgrading.  Thus, we require 
    //  finding 2 periods, not just one, after the intrinsic name.
    if (Name.compare(5,9,"part.set.",9) == 0) {
      std::string::size_type delim = Name.find('.',14);

      if (delim != std::string::npos &&
          Name.find('.',delim+1) != std::string::npos) {
        //  Construct a new name as 'llvm.part.select' + '.i*.i*'
        F->setName(Name.substr(0,13)+Name.substr(delim));
        NewFn = F;
        return true;
      }
      break;
    }

    break;
  case 'x': 
    // This fixes all MMX shift intrinsic instructions to take a
    // v1i64 instead of a v2i32 as the second parameter.
    if (Name.compare(5,10,"x86.mmx.ps",10) == 0 &&
        (Name.compare(13,4,"psll", 4) == 0 ||
         Name.compare(13,4,"psra", 4) == 0 ||
         Name.compare(13,4,"psrl", 4) == 0) && Name[17] != 'i') {
      
      const llvm::Type *VT =
                    VectorType::get(IntegerType::get(FTy->getContext(), 64), 1);
      
      // We don't have to do anything if the parameter already has
      // the correct type.
      if (FTy->getParamType(1) == VT)
        break;
      
      //  We first need to change the name of the old (bad) intrinsic, because 
      //  its type is incorrect, but we cannot overload that name. We 
      //  arbitrarily unique it here allowing us to construct a correctly named 
      //  and typed function below.
      F->setName("");

      assert(FTy->getNumParams() == 2 && "MMX shift intrinsics take 2 args!");
      
      //  Now construct the new intrinsic with the correct name and type. We 
      //  leave the old function around in order to query its type, whatever it 
      //  may be, and correctly convert up to the new type.
      NewFn = cast<Function>(M->getOrInsertFunction(Name, 
                                                    FTy->getReturnType(),
                                                    FTy->getParamType(0),
                                                    VT,
                                                    (Type *)0));
      return true;
    } else if (Name.compare(5,17,"x86.sse2.loadh.pd",17) == 0 ||
               Name.compare(5,17,"x86.sse2.loadl.pd",17) == 0 ||
               Name.compare(5,16,"x86.sse2.movl.dq",16) == 0 ||
               Name.compare(5,15,"x86.sse2.movs.d",15) == 0 ||
               Name.compare(5,16,"x86.sse2.shuf.pd",16) == 0 ||
               Name.compare(5,18,"x86.sse2.unpckh.pd",18) == 0 ||
               Name.compare(5,18,"x86.sse2.unpckl.pd",18) == 0 ||
               Name.compare(5,20,"x86.sse2.punpckh.qdq",20) == 0 ||
               Name.compare(5,20,"x86.sse2.punpckl.qdq",20) == 0) {
      // Calls to these intrinsics are transformed into ShuffleVector's.
      NewFn = 0;
      return true;
    } else if (Name.compare(5, 16, "x86.sse41.pmulld", 16) == 0) {
      // Calls to these intrinsics are transformed into vector multiplies.
      NewFn = 0;
      return true;
    } else if (Name.compare(5, 18, "x86.ssse3.palign.r", 18) == 0 ||
               Name.compare(5, 22, "x86.ssse3.palign.r.128", 22) == 0) {
      // Calls to these intrinsics are transformed into vector shuffles, shifts,
      // or 0.
      NewFn = 0;
      return true;           
    }

    break;
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
    F->setAttributes(Intrinsic::getAttributes((Intrinsic::ID)id));
  return Upgraded;
}

bool llvm::UpgradeGlobalVariable(GlobalVariable *GV) {
  StringRef Name(GV->getName());

  // We are only upgrading one symbol here.
  if (Name == ".llvm.eh.catch.all.value") {
    GV->setName("llvm.eh.catch.all.value");
    return true;
  }

  return false;
}

/// ExtendNEONArgs - For NEON "long" and "wide" operations, where the results
/// have vector elements twice as big as one or both source operands, do the
/// sign- or zero-extension that used to be handled by intrinsics.  The
/// extended values are returned via V0 and V1.
static void ExtendNEONArgs(CallInst *CI, Value *Arg0, Value *Arg1,
                           Value *&V0, Value *&V1) {
  Function *F = CI->getCalledFunction();
  const std::string& Name = F->getName();
  bool isLong = (Name.at(18) == 'l');
  bool isSigned = (Name.at(19) == 's');

  if (isSigned) {
    if (isLong)
      V0 = new SExtInst(Arg0, CI->getType(), "", CI);
    else
      V0 = Arg0;
    V1 = new SExtInst(Arg1, CI->getType(), "", CI);
  } else {
    if (isLong)
      V0 = new ZExtInst(Arg0, CI->getType(), "", CI);
    else
      V0 = Arg0;
    V1 = new ZExtInst(Arg1, CI->getType(), "", CI);
  }
}

/// CallVABD - As part of expanding a call to one of the old NEON vabdl, vaba,
/// or vabal intrinsics, construct a call to a vabd intrinsic.  Examine the
/// name of the old intrinsic to determine whether to use a signed or unsigned
/// vabd intrinsic.  Get the type from the old call instruction, adjusted for
/// half-size vector elements if the old intrinsic was vabdl or vabal.
static Instruction *CallVABD(CallInst *CI, Value *Arg0, Value *Arg1) {
  Function *F = CI->getCalledFunction();
  const std::string& Name = F->getName();
  bool isLong = (Name.at(18) == 'l');
  bool isSigned = (Name.at(isLong ? 19 : 18) == 's');

  Intrinsic::ID intID;
  if (isSigned)
    intID = Intrinsic::arm_neon_vabds;
  else
    intID = Intrinsic::arm_neon_vabdu;

  const Type *Ty = CI->getType();
  if (isLong)
    Ty = VectorType::getTruncatedElementVectorType(cast<const VectorType>(Ty));

  Function *VABD = Intrinsic::getDeclaration(F->getParent(), intID, &Ty, 1);
  Value *Operands[2];
  Operands[0] = Arg0;
  Operands[1] = Arg1;
  return CallInst::Create(VABD, Operands, Operands+2, 
                          "upgraded."+CI->getName(), CI);
}

// UpgradeIntrinsicCall - Upgrade a call to an old intrinsic to be a call the 
// upgraded intrinsic. All argument and return casting must be provided in 
// order to seamlessly integrate with existing context.
void llvm::UpgradeIntrinsicCall(CallInst *CI, Function *NewFn) {
  Function *F = CI->getCalledFunction();
  LLVMContext &C = CI->getContext();
  ImmutableCallSite CS(CI);

  assert(F && "CallInst has no function associated with it.");

  if (!NewFn) {
    // Get the Function's name.
    const std::string& Name = F->getName();

    // Upgrade ARM NEON intrinsics.
    if (Name.compare(5, 9, "arm.neon.", 9) == 0) {
      Instruction *NewI;
      Value *V0, *V1;
      if (Name.compare(14, 7, "vmovls.", 7) == 0) {
        NewI = new SExtInst(CI->getArgOperand(0), CI->getType(),
                            "upgraded." + CI->getName(), CI);
      } else if (Name.compare(14, 7, "vmovlu.", 7) == 0) {
        NewI = new ZExtInst(CI->getArgOperand(0), CI->getType(),
                            "upgraded." + CI->getName(), CI);
      } else if (Name.compare(14, 4, "vadd", 4) == 0) {
        ExtendNEONArgs(CI, CI->getArgOperand(0), CI->getArgOperand(1), V0, V1);
        NewI = BinaryOperator::CreateAdd(V0, V1, "upgraded."+CI->getName(), CI);
      } else if (Name.compare(14, 4, "vsub", 4) == 0) {
        ExtendNEONArgs(CI, CI->getArgOperand(0), CI->getArgOperand(1), V0, V1);
        NewI = BinaryOperator::CreateSub(V0, V1,"upgraded."+CI->getName(),CI);
      } else if (Name.compare(14, 4, "vmul", 4) == 0) {
        ExtendNEONArgs(CI, CI->getArgOperand(0), CI->getArgOperand(1), V0, V1);
        NewI = BinaryOperator::CreateMul(V0, V1,"upgraded."+CI->getName(),CI);
      } else if (Name.compare(14, 4, "vmla", 4) == 0) {
        ExtendNEONArgs(CI, CI->getArgOperand(1), CI->getArgOperand(2), V0, V1);
        Instruction *MulI = BinaryOperator::CreateMul(V0, V1, "", CI);
        NewI = BinaryOperator::CreateAdd(CI->getArgOperand(0), MulI,
                                         "upgraded."+CI->getName(), CI);
      } else if (Name.compare(14, 4, "vmls", 4) == 0) {
        ExtendNEONArgs(CI, CI->getArgOperand(1), CI->getArgOperand(2), V0, V1);
        Instruction *MulI = BinaryOperator::CreateMul(V0, V1, "", CI);
        NewI = BinaryOperator::CreateSub(CI->getArgOperand(0), MulI,
                                         "upgraded."+CI->getName(), CI);
      } else if (Name.compare(14, 4, "vabd", 4) == 0) {
        NewI = CallVABD(CI, CI->getArgOperand(0), CI->getArgOperand(1));
        NewI = new ZExtInst(NewI, CI->getType(), "upgraded."+CI->getName(), CI);
      } else if (Name.compare(14, 4, "vaba", 4) == 0) {
        NewI = CallVABD(CI, CI->getArgOperand(1), CI->getArgOperand(2));
        if (Name.at(18) == 'l')
          NewI = new ZExtInst(NewI, CI->getType(), "", CI);
        NewI = BinaryOperator::CreateAdd(CI->getArgOperand(0), NewI,
                                         "upgraded."+CI->getName(), CI);
      } else if (Name.compare(14, 6, "vmovn.", 6) == 0) {
        NewI = new TruncInst(CI->getArgOperand(0), CI->getType(),
                             "upgraded." + CI->getName(), CI);
      } else {
        llvm_unreachable("Unknown arm.neon function for CallInst upgrade.");
      }
      // Replace any uses of the old CallInst.
      if (!CI->use_empty())
        CI->replaceAllUsesWith(NewI);
      CI->eraseFromParent();
      return;
    }

    bool isLoadH = false, isLoadL = false, isMovL = false;
    bool isMovSD = false, isShufPD = false;
    bool isUnpckhPD = false, isUnpcklPD = false;
    bool isPunpckhQPD = false, isPunpcklQPD = false;
    if (F->getName() == "llvm.x86.sse2.loadh.pd")
      isLoadH = true;
    else if (F->getName() == "llvm.x86.sse2.loadl.pd")
      isLoadL = true;
    else if (F->getName() == "llvm.x86.sse2.movl.dq")
      isMovL = true;
    else if (F->getName() == "llvm.x86.sse2.movs.d")
      isMovSD = true;
    else if (F->getName() == "llvm.x86.sse2.shuf.pd")
      isShufPD = true;
    else if (F->getName() == "llvm.x86.sse2.unpckh.pd")
      isUnpckhPD = true;
    else if (F->getName() == "llvm.x86.sse2.unpckl.pd")
      isUnpcklPD = true;
    else if (F->getName() ==  "llvm.x86.sse2.punpckh.qdq")
      isPunpckhQPD = true;
    else if (F->getName() ==  "llvm.x86.sse2.punpckl.qdq")
      isPunpcklQPD = true;

    if (isLoadH || isLoadL || isMovL || isMovSD || isShufPD ||
        isUnpckhPD || isUnpcklPD || isPunpckhQPD || isPunpcklQPD) {
      std::vector<Constant*> Idxs;
      Value *Op0 = CI->getArgOperand(0);
      ShuffleVectorInst *SI = NULL;
      if (isLoadH || isLoadL) {
        Value *Op1 = UndefValue::get(Op0->getType());
        Value *Addr = new BitCastInst(CI->getArgOperand(1), 
                                  Type::getDoublePtrTy(C),
                                      "upgraded.", CI);
        Value *Load = new LoadInst(Addr, "upgraded.", false, 8, CI);
        Value *Idx = ConstantInt::get(Type::getInt32Ty(C), 0);
        Op1 = InsertElementInst::Create(Op1, Load, Idx, "upgraded.", CI);

        if (isLoadH) {
          Idxs.push_back(ConstantInt::get(Type::getInt32Ty(C), 0));
          Idxs.push_back(ConstantInt::get(Type::getInt32Ty(C), 2));
        } else {
          Idxs.push_back(ConstantInt::get(Type::getInt32Ty(C), 2));
          Idxs.push_back(ConstantInt::get(Type::getInt32Ty(C), 1));
        }
        Value *Mask = ConstantVector::get(Idxs);
        SI = new ShuffleVectorInst(Op0, Op1, Mask, "upgraded.", CI);
      } else if (isMovL) {
        Constant *Zero = ConstantInt::get(Type::getInt32Ty(C), 0);
        Idxs.push_back(Zero);
        Idxs.push_back(Zero);
        Idxs.push_back(Zero);
        Idxs.push_back(Zero);
        Value *ZeroV = ConstantVector::get(Idxs);

        Idxs.clear(); 
        Idxs.push_back(ConstantInt::get(Type::getInt32Ty(C), 4));
        Idxs.push_back(ConstantInt::get(Type::getInt32Ty(C), 5));
        Idxs.push_back(ConstantInt::get(Type::getInt32Ty(C), 2));
        Idxs.push_back(ConstantInt::get(Type::getInt32Ty(C), 3));
        Value *Mask = ConstantVector::get(Idxs);
        SI = new ShuffleVectorInst(ZeroV, Op0, Mask, "upgraded.", CI);
      } else if (isMovSD ||
                 isUnpckhPD || isUnpcklPD || isPunpckhQPD || isPunpcklQPD) {
        Value *Op1 = CI->getArgOperand(1);
        if (isMovSD) {
          Idxs.push_back(ConstantInt::get(Type::getInt32Ty(C), 2));
          Idxs.push_back(ConstantInt::get(Type::getInt32Ty(C), 1));
        } else if (isUnpckhPD || isPunpckhQPD) {
          Idxs.push_back(ConstantInt::get(Type::getInt32Ty(C), 1));
          Idxs.push_back(ConstantInt::get(Type::getInt32Ty(C), 3));
        } else {
          Idxs.push_back(ConstantInt::get(Type::getInt32Ty(C), 0));
          Idxs.push_back(ConstantInt::get(Type::getInt32Ty(C), 2));
        }
        Value *Mask = ConstantVector::get(Idxs);
        SI = new ShuffleVectorInst(Op0, Op1, Mask, "upgraded.", CI);
      } else if (isShufPD) {
        Value *Op1 = CI->getArgOperand(1);
        unsigned MaskVal =
                        cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue();
        Idxs.push_back(ConstantInt::get(Type::getInt32Ty(C), MaskVal & 1));
        Idxs.push_back(ConstantInt::get(Type::getInt32Ty(C),
                                               ((MaskVal >> 1) & 1)+2));
        Value *Mask = ConstantVector::get(Idxs);
        SI = new ShuffleVectorInst(Op0, Op1, Mask, "upgraded.", CI);
      }

      assert(SI && "Unexpected!");

      // Handle any uses of the old CallInst.
      if (!CI->use_empty())
        //  Replace all uses of the old call with the new cast which has the 
        //  correct type.
        CI->replaceAllUsesWith(SI);
      
      //  Clean up the old call now that it has been completely upgraded.
      CI->eraseFromParent();
    } else if (F->getName() == "llvm.x86.sse41.pmulld") {
      // Upgrade this set of intrinsics into vector multiplies.
      Instruction *Mul = BinaryOperator::CreateMul(CI->getArgOperand(0),
                                                   CI->getArgOperand(1),
                                                   CI->getName(),
                                                   CI);
      // Fix up all the uses with our new multiply.
      if (!CI->use_empty())
        CI->replaceAllUsesWith(Mul);
        
      // Remove upgraded multiply.
      CI->eraseFromParent();
    } else if (F->getName() == "llvm.x86.ssse3.palign.r") {
      Value *Op1 = CI->getArgOperand(0);
      Value *Op2 = CI->getArgOperand(1);
      Value *Op3 = CI->getArgOperand(2);
      unsigned shiftVal = cast<ConstantInt>(Op3)->getZExtValue();
      Value *Rep;
      IRBuilder<> Builder(C);
      Builder.SetInsertPoint(CI->getParent(), CI);

      // If palignr is shifting the pair of input vectors less than 9 bytes,
      // emit a shuffle instruction.
      if (shiftVal <= 8) {
        const Type *IntTy = Type::getInt32Ty(C);
        const Type *EltTy = Type::getInt8Ty(C);
        const Type *VecTy = VectorType::get(EltTy, 8);
        
        Op2 = Builder.CreateBitCast(Op2, VecTy);
        Op1 = Builder.CreateBitCast(Op1, VecTy);

        llvm::SmallVector<llvm::Constant*, 8> Indices;
        for (unsigned i = 0; i != 8; ++i)
          Indices.push_back(ConstantInt::get(IntTy, shiftVal + i));

        Value *SV = ConstantVector::get(Indices.begin(), Indices.size());
        Rep = Builder.CreateShuffleVector(Op2, Op1, SV, "palignr");
        Rep = Builder.CreateBitCast(Rep, F->getReturnType());
      }

      // If palignr is shifting the pair of input vectors more than 8 but less
      // than 16 bytes, emit a logical right shift of the destination.
      else if (shiftVal < 16) {
        // MMX has these as 1 x i64 vectors for some odd optimization reasons.
        const Type *EltTy = Type::getInt64Ty(C);
        const Type *VecTy = VectorType::get(EltTy, 1);

        Op1 = Builder.CreateBitCast(Op1, VecTy, "cast");
        Op2 = ConstantInt::get(VecTy, (shiftVal-8) * 8);

        // create i32 constant
        Function *I =
          Intrinsic::getDeclaration(F->getParent(), Intrinsic::x86_mmx_psrl_q);
        Rep = Builder.CreateCall2(I, Op1, Op2, "palignr");
      }

      // If palignr is shifting the pair of vectors more than 32 bytes, emit zero.
      else {
        Rep = Constant::getNullValue(F->getReturnType());
      }
      
      // Replace any uses with our new instruction.
      if (!CI->use_empty())
        CI->replaceAllUsesWith(Rep);
        
      // Remove upgraded instruction.
      CI->eraseFromParent();
      
    } else if (F->getName() == "llvm.x86.ssse3.palign.r.128") {
      Value *Op1 = CI->getArgOperand(0);
      Value *Op2 = CI->getArgOperand(1);
      Value *Op3 = CI->getArgOperand(2);
      unsigned shiftVal = cast<ConstantInt>(Op3)->getZExtValue();
      Value *Rep;
      IRBuilder<> Builder(C);
      Builder.SetInsertPoint(CI->getParent(), CI);

      // If palignr is shifting the pair of input vectors less than 17 bytes,
      // emit a shuffle instruction.
      if (shiftVal <= 16) {
        const Type *IntTy = Type::getInt32Ty(C);
        const Type *EltTy = Type::getInt8Ty(C);
        const Type *VecTy = VectorType::get(EltTy, 16);
        
        Op2 = Builder.CreateBitCast(Op2, VecTy);
        Op1 = Builder.CreateBitCast(Op1, VecTy);

        llvm::SmallVector<llvm::Constant*, 16> Indices;
        for (unsigned i = 0; i != 16; ++i)
          Indices.push_back(ConstantInt::get(IntTy, shiftVal + i));

        Value *SV = ConstantVector::get(Indices.begin(), Indices.size());
        Rep = Builder.CreateShuffleVector(Op2, Op1, SV, "palignr");
        Rep = Builder.CreateBitCast(Rep, F->getReturnType());
      }

      // If palignr is shifting the pair of input vectors more than 16 but less
      // than 32 bytes, emit a logical right shift of the destination.
      else if (shiftVal < 32) {
        const Type *EltTy = Type::getInt64Ty(C);
        const Type *VecTy = VectorType::get(EltTy, 2);
        const Type *IntTy = Type::getInt32Ty(C);

        Op1 = Builder.CreateBitCast(Op1, VecTy, "cast");
        Op2 = ConstantInt::get(IntTy, (shiftVal-16) * 8);

        // create i32 constant
        Function *I =
          Intrinsic::getDeclaration(F->getParent(), Intrinsic::x86_sse2_psrl_dq);
        Rep = Builder.CreateCall2(I, Op1, Op2, "palignr");
      }

      // If palignr is shifting the pair of vectors more than 32 bytes, emit zero.
      else {
        Rep = Constant::getNullValue(F->getReturnType());
      }
      
      // Replace any uses with our new instruction.
      if (!CI->use_empty())
        CI->replaceAllUsesWith(Rep);
        
      // Remove upgraded instruction.
      CI->eraseFromParent();
      
    } else {
      llvm_unreachable("Unknown function for CallInst upgrade.");
    }
    return;
  }

  switch (NewFn->getIntrinsicID()) {
  default: llvm_unreachable("Unknown function for CallInst upgrade.");
  case Intrinsic::arm_neon_vld1:
  case Intrinsic::arm_neon_vld2:
  case Intrinsic::arm_neon_vld3:
  case Intrinsic::arm_neon_vld4:
  case Intrinsic::arm_neon_vst1:
  case Intrinsic::arm_neon_vst2:
  case Intrinsic::arm_neon_vst3:
  case Intrinsic::arm_neon_vst4:
  case Intrinsic::arm_neon_vld2lane:
  case Intrinsic::arm_neon_vld3lane:
  case Intrinsic::arm_neon_vld4lane:
  case Intrinsic::arm_neon_vst2lane:
  case Intrinsic::arm_neon_vst3lane:
  case Intrinsic::arm_neon_vst4lane: {
    // Add a default alignment argument of 1.
    SmallVector<Value*, 8> Operands(CS.arg_begin(), CS.arg_end());
    Operands.push_back(ConstantInt::get(Type::getInt32Ty(C), 1));
    CallInst *NewCI = CallInst::Create(NewFn, Operands.begin(), Operands.end(),
                                       CI->getName(), CI);
    NewCI->setTailCall(CI->isTailCall());
    NewCI->setCallingConv(CI->getCallingConv());

    //  Handle any uses of the old CallInst.
    if (!CI->use_empty())
      //  Replace all uses of the old call with the new cast which has the 
      //  correct type.
      CI->replaceAllUsesWith(NewCI);
    
    //  Clean up the old call now that it has been completely upgraded.
    CI->eraseFromParent();
    break;
  }        

  case Intrinsic::x86_mmx_psll_d:
  case Intrinsic::x86_mmx_psll_q:
  case Intrinsic::x86_mmx_psll_w:
  case Intrinsic::x86_mmx_psra_d:
  case Intrinsic::x86_mmx_psra_w:
  case Intrinsic::x86_mmx_psrl_d:
  case Intrinsic::x86_mmx_psrl_q:
  case Intrinsic::x86_mmx_psrl_w: {
    Value *Operands[2];
    
    Operands[0] = CI->getArgOperand(0);
    
    // Cast the second parameter to the correct type.
    BitCastInst *BC = new BitCastInst(CI->getArgOperand(1), 
                                      NewFn->getFunctionType()->getParamType(1),
                                      "upgraded.", CI);
    Operands[1] = BC;
    
    //  Construct a new CallInst
    CallInst *NewCI = CallInst::Create(NewFn, Operands, Operands+2, 
                                       "upgraded."+CI->getName(), CI);
    NewCI->setTailCall(CI->isTailCall());
    NewCI->setCallingConv(CI->getCallingConv());
    
    //  Handle any uses of the old CallInst.
    if (!CI->use_empty())
      //  Replace all uses of the old call with the new cast which has the 
      //  correct type.
      CI->replaceAllUsesWith(NewCI);
    
    //  Clean up the old call now that it has been completely upgraded.
    CI->eraseFromParent();
    break;
  }        
  case Intrinsic::ctlz:
  case Intrinsic::ctpop:
  case Intrinsic::cttz: {
    //  Build a small vector of the original arguments.
    SmallVector<Value*, 8> Operands(CS.arg_begin(), CS.arg_end());

    //  Construct a new CallInst
    CallInst *NewCI = CallInst::Create(NewFn, Operands.begin(), Operands.end(),
                                       "upgraded."+CI->getName(), CI);
    NewCI->setTailCall(CI->isTailCall());
    NewCI->setCallingConv(CI->getCallingConv());

    //  Handle any uses of the old CallInst.
    if (!CI->use_empty()) {
      //  Check for sign extend parameter attributes on the return values.
      bool SrcSExt = NewFn->getAttributes().paramHasAttr(0, Attribute::SExt);
      bool DestSExt = F->getAttributes().paramHasAttr(0, Attribute::SExt);
      
      //  Construct an appropriate cast from the new return type to the old.
      CastInst *RetCast = CastInst::Create(
                            CastInst::getCastOpcode(NewCI, SrcSExt,
                                                    F->getReturnType(),
                                                    DestSExt),
                            NewCI, F->getReturnType(),
                            NewCI->getName(), CI);
      NewCI->moveBefore(RetCast);

      //  Replace all uses of the old call with the new cast which has the 
      //  correct type.
      CI->replaceAllUsesWith(RetCast);
    }

    //  Clean up the old call now that it has been completely upgraded.
    CI->eraseFromParent();
  }
  break;
  case Intrinsic::eh_selector:
  case Intrinsic::eh_typeid_for: {
    // Only the return type changed.
    SmallVector<Value*, 8> Operands(CS.arg_begin(), CS.arg_end());
    CallInst *NewCI = CallInst::Create(NewFn, Operands.begin(), Operands.end(),
                                       "upgraded." + CI->getName(), CI);
    NewCI->setTailCall(CI->isTailCall());
    NewCI->setCallingConv(CI->getCallingConv());

    //  Handle any uses of the old CallInst.
    if (!CI->use_empty()) {
      //  Construct an appropriate cast from the new return type to the old.
      CastInst *RetCast =
        CastInst::Create(CastInst::getCastOpcode(NewCI, true,
                                                 F->getReturnType(), true),
                         NewCI, F->getReturnType(), NewCI->getName(), CI);
      CI->replaceAllUsesWith(RetCast);
    }
    CI->eraseFromParent();
  }
  break;
  case Intrinsic::memcpy:
  case Intrinsic::memmove:
  case Intrinsic::memset: {
    // Add isVolatile
    const llvm::Type *I1Ty = llvm::Type::getInt1Ty(CI->getContext());
    Value *Operands[5] = { CI->getArgOperand(0), CI->getArgOperand(1),
                           CI->getArgOperand(2), CI->getArgOperand(3),
                           llvm::ConstantInt::get(I1Ty, 0) };
    CallInst *NewCI = CallInst::Create(NewFn, Operands, Operands+5,
                                       CI->getName(), CI);
    NewCI->setTailCall(CI->isTailCall());
    NewCI->setCallingConv(CI->getCallingConv());
    //  Handle any uses of the old CallInst.
    if (!CI->use_empty())
      //  Replace all uses of the old call with the new cast which has the 
      //  correct type.
      CI->replaceAllUsesWith(NewCI);
    
    //  Clean up the old call now that it has been completely upgraded.
    CI->eraseFromParent();
    break;
  }
  }
}

// This tests each Function to determine if it needs upgrading. When we find 
// one we are interested in, we then upgrade all calls to reflect the new 
// function.
void llvm::UpgradeCallsToIntrinsic(Function* F) {
  assert(F && "Illegal attempt to upgrade a non-existent intrinsic.");

  // Upgrade the function and check if it is a totaly new function.
  Function* NewFn;
  if (UpgradeIntrinsicFunction(F, NewFn)) {
    if (NewFn != F) {
      // Replace all uses to the old function with the new one if necessary.
      for (Value::use_iterator UI = F->use_begin(), UE = F->use_end();
           UI != UE; ) {
        if (CallInst* CI = dyn_cast<CallInst>(*UI++))
          UpgradeIntrinsicCall(CI, NewFn);
      }
      // Remove old function, no longer used, from the module.
      F->eraseFromParent();
    }
  }
}

/// This function strips all debug info intrinsics, except for llvm.dbg.declare.
/// If an llvm.dbg.declare intrinsic is invalid, then this function simply
/// strips that use.
void llvm::CheckDebugInfoIntrinsics(Module *M) {


  if (Function *FuncStart = M->getFunction("llvm.dbg.func.start")) {
    while (!FuncStart->use_empty()) {
      CallInst *CI = cast<CallInst>(FuncStart->use_back());
      CI->eraseFromParent();
    }
    FuncStart->eraseFromParent();
  }
  
  if (Function *StopPoint = M->getFunction("llvm.dbg.stoppoint")) {
    while (!StopPoint->use_empty()) {
      CallInst *CI = cast<CallInst>(StopPoint->use_back());
      CI->eraseFromParent();
    }
    StopPoint->eraseFromParent();
  }

  if (Function *RegionStart = M->getFunction("llvm.dbg.region.start")) {
    while (!RegionStart->use_empty()) {
      CallInst *CI = cast<CallInst>(RegionStart->use_back());
      CI->eraseFromParent();
    }
    RegionStart->eraseFromParent();
  }

  if (Function *RegionEnd = M->getFunction("llvm.dbg.region.end")) {
    while (!RegionEnd->use_empty()) {
      CallInst *CI = cast<CallInst>(RegionEnd->use_back());
      CI->eraseFromParent();
    }
    RegionEnd->eraseFromParent();
  }
  
  if (Function *Declare = M->getFunction("llvm.dbg.declare")) {
    if (!Declare->use_empty()) {
      DbgDeclareInst *DDI = cast<DbgDeclareInst>(Declare->use_back());
      if (!isa<MDNode>(DDI->getArgOperand(0)) ||
          !isa<MDNode>(DDI->getArgOperand(1))) {
        while (!Declare->use_empty()) {
          CallInst *CI = cast<CallInst>(Declare->use_back());
          CI->eraseFromParent();
        }
        Declare->eraseFromParent();
      }
    }
  }
}
