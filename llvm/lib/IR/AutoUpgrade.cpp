//===-- AutoUpgrade.cpp - Implement auto-upgrade helper functions ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the auto-upgrade helper functions.
// This is where deprecated IR intrinsics and other IR features are updated to
// current specifications.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Regex.h"
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

// Upgrade the declarations of intrinsic functions whose 8-bit immediate mask
// arguments have changed their type from i32 to i8.
static bool UpgradeX86IntrinsicsWith8BitMask(Function *F, Intrinsic::ID IID,
                                             Function *&NewFn) {
  // Check that the last argument is an i32.
  Type *LastArgType = F->getFunctionType()->getParamType(
     F->getFunctionType()->getNumParams() - 1);
  if (!LastArgType->isIntegerTy(32))
    return false;

  // Move this function aside and map down.
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
    Regex vldRegex("^arm\\.neon\\.vld([1234]|[234]lane)\\.v[a-z0-9]*$");
    if (vldRegex.match(Name)) {
      auto fArgs = F->getFunctionType()->params();
      SmallVector<Type *, 4> Tys(fArgs.begin(), fArgs.end());
      // Can't use Intrinsic::getDeclaration here as the return types might
      // then only be structurally equal.
      FunctionType* fType = FunctionType::get(F->getReturnType(), Tys, false);
      NewFn = Function::Create(fType, F->getLinkage(),
                               "llvm." + Name + ".p0i8", F->getParent());
      return true;
    }
    Regex vstRegex("^arm\\.neon\\.vst([1234]|[234]lane)\\.v[a-z0-9]*$");
    if (vstRegex.match(Name)) {
      static const Intrinsic::ID StoreInts[] = {Intrinsic::arm_neon_vst1,
                                                Intrinsic::arm_neon_vst2,
                                                Intrinsic::arm_neon_vst3,
                                                Intrinsic::arm_neon_vst4};

      static const Intrinsic::ID StoreLaneInts[] = {
        Intrinsic::arm_neon_vst2lane, Intrinsic::arm_neon_vst3lane,
        Intrinsic::arm_neon_vst4lane
      };

      auto fArgs = F->getFunctionType()->params();
      Type *Tys[] = {fArgs[0], fArgs[1]};
      if (Name.find("lane") == StringRef::npos)
        NewFn = Intrinsic::getDeclaration(F->getParent(),
                                          StoreInts[fArgs.size() - 3], Tys);
      else
        NewFn = Intrinsic::getDeclaration(F->getParent(),
                                          StoreLaneInts[fArgs.size() - 5], Tys);
      return true;
    }
    if (Name == "aarch64.thread.pointer" || Name == "arm.thread.pointer") {
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::thread_pointer);
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

  case 'm': {
    if (Name.startswith("masked.load.")) {
      Type *Tys[] = { F->getReturnType(), F->arg_begin()->getType() };
      if (F->getName() != Intrinsic::getName(Intrinsic::masked_load, Tys)) {
        F->setName(Name + ".old");
        NewFn = Intrinsic::getDeclaration(F->getParent(),
                                          Intrinsic::masked_load,
                                          Tys);
        return true;
      }
    }
    if (Name.startswith("masked.store.")) {
      auto Args = F->getFunctionType()->params();
      Type *Tys[] = { Args[0], Args[1] };
      if (F->getName() != Intrinsic::getName(Intrinsic::masked_store, Tys)) {
        F->setName(Name + ".old");
        NewFn = Intrinsic::getDeclaration(F->getParent(),
                                          Intrinsic::masked_store,
                                          Tys);
        return true;
      }
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

  case 's':
    if (Name == "stackprotectorcheck") {
      NewFn = nullptr;
      return true;
    }

  case 'x': {
    bool IsX86 = Name.startswith("x86.");
    if (IsX86)
      Name = Name.substr(4);

    if (IsX86 &&
        (Name.startswith("sse2.pcmpeq.") ||
         Name.startswith("sse2.pcmpgt.") ||
         Name.startswith("avx2.pcmpeq.") ||
         Name.startswith("avx2.pcmpgt.") ||
         Name.startswith("avx512.mask.pcmpeq.") ||
         Name.startswith("avx512.mask.pcmpgt.") ||
         Name == "sse41.pmaxsb" ||
         Name == "sse2.pmaxs.w" ||
         Name == "sse41.pmaxsd" ||
         Name == "sse2.pmaxu.b" ||
         Name == "sse41.pmaxuw" ||
         Name == "sse41.pmaxud" ||
         Name == "sse41.pminsb" ||
         Name == "sse2.pmins.w" ||
         Name == "sse41.pminsd" ||
         Name == "sse2.pminu.b" ||
         Name == "sse41.pminuw" ||
         Name == "sse41.pminud" ||
         Name.startswith("avx2.pmax") ||
         Name.startswith("avx2.pmin") ||
         Name.startswith("avx2.vbroadcast") ||
         Name.startswith("avx2.pbroadcast") ||
         Name.startswith("avx.vpermil.") ||
         Name.startswith("sse2.pshuf") ||
         Name.startswith("avx512.pbroadcast") ||
         Name.startswith("avx512.mask.broadcast.s") ||
         Name.startswith("avx512.mask.movddup") ||
         Name.startswith("avx512.mask.movshdup") ||
         Name.startswith("avx512.mask.movsldup") ||
         Name.startswith("avx512.mask.pshuf.d.") ||
         Name.startswith("avx512.mask.pshufl.w.") ||
         Name.startswith("avx512.mask.pshufh.w.") ||
         Name.startswith("avx512.mask.vpermil.p") ||
         Name.startswith("avx512.mask.perm.df.") ||
         Name.startswith("avx512.mask.perm.di.") ||
         Name.startswith("avx512.mask.punpckl") ||
         Name.startswith("avx512.mask.punpckh") ||
         Name.startswith("avx512.mask.unpckl.") ||
         Name.startswith("avx512.mask.unpckh.") ||
         Name.startswith("avx512.mask.pand.") ||
         Name.startswith("avx512.mask.pandn.") ||
         Name.startswith("avx512.mask.por.") ||
         Name.startswith("avx512.mask.pxor.") ||
         Name.startswith("sse41.pmovsx") ||
         Name.startswith("sse41.pmovzx") ||
         Name.startswith("avx2.pmovsx") ||
         Name.startswith("avx2.pmovzx") ||
         Name == "sse2.cvtdq2pd" ||
         Name == "sse2.cvtps2pd" ||
         Name == "avx.cvtdq2.pd.256" ||
         Name == "avx.cvt.ps2.pd.256" ||
         Name.startswith("avx.vinsertf128.") ||
         Name == "avx2.vinserti128" ||
         Name.startswith("avx.vextractf128.") ||
         Name == "avx2.vextracti128" ||
         Name.startswith("sse4a.movnt.") ||
         Name.startswith("avx.movnt.") ||
         Name.startswith("avx512.storent.") ||
         Name == "sse2.storel.dq" ||
         Name.startswith("sse.storeu.") ||
         Name.startswith("sse2.storeu.") ||
         Name.startswith("avx.storeu.") ||
         Name.startswith("avx512.mask.storeu.p") ||
         Name.startswith("avx512.mask.storeu.b.") ||
         Name.startswith("avx512.mask.storeu.w.") ||
         Name.startswith("avx512.mask.storeu.d.") ||
         Name.startswith("avx512.mask.storeu.q.") ||
         Name.startswith("avx512.mask.store.p") ||
         Name.startswith("avx512.mask.store.b.") ||
         Name.startswith("avx512.mask.store.w.") ||
         Name.startswith("avx512.mask.store.d.") ||
         Name.startswith("avx512.mask.store.q.") ||
         Name.startswith("avx512.mask.loadu.p") ||
         Name.startswith("avx512.mask.loadu.b.") ||
         Name.startswith("avx512.mask.loadu.w.") ||
         Name.startswith("avx512.mask.loadu.d.") ||
         Name.startswith("avx512.mask.loadu.q.") ||
         Name.startswith("avx512.mask.load.p") ||
         Name.startswith("avx512.mask.load.b.") ||
         Name.startswith("avx512.mask.load.w.") ||
         Name.startswith("avx512.mask.load.d.") ||
         Name.startswith("avx512.mask.load.q.") ||
         Name == "sse42.crc32.64.8" ||
         Name.startswith("avx.vbroadcast.s") ||
         Name.startswith("avx512.mask.palignr.") ||
         Name.startswith("sse2.psll.dq") ||
         Name.startswith("sse2.psrl.dq") ||
         Name.startswith("avx2.psll.dq") ||
         Name.startswith("avx2.psrl.dq") ||
         Name.startswith("avx512.psll.dq") ||
         Name.startswith("avx512.psrl.dq") ||
         Name == "sse41.pblendw" ||
         Name.startswith("sse41.blendp") ||
         Name.startswith("avx.blend.p") ||
         Name == "avx2.pblendw" ||
         Name.startswith("avx2.pblendd.") ||
         Name == "avx2.vbroadcasti128" ||
         Name == "xop.vpcmov" ||
         (Name.startswith("xop.vpcom") && F->arg_size() == 2))) {
      NewFn = nullptr;
      return true;
    }
    // SSE4.1 ptest functions may have an old signature.
    if (IsX86 && Name.startswith("sse41.ptest")) {
      if (Name.substr(11) == "c")
        return UpgradeSSE41Function(F, Intrinsic::x86_sse41_ptestc, NewFn);
      if (Name.substr(11) == "z")
        return UpgradeSSE41Function(F, Intrinsic::x86_sse41_ptestz, NewFn);
      if (Name.substr(11) == "nzc")
        return UpgradeSSE41Function(F, Intrinsic::x86_sse41_ptestnzc, NewFn);
    }
    // Several blend and other instructions with masks used the wrong number of
    // bits.
    if (IsX86 && Name == "sse41.insertps")
      return UpgradeX86IntrinsicsWith8BitMask(F, Intrinsic::x86_sse41_insertps,
                                              NewFn);
    if (IsX86 && Name == "sse41.dppd")
      return UpgradeX86IntrinsicsWith8BitMask(F, Intrinsic::x86_sse41_dppd,
                                              NewFn);
    if (IsX86 && Name == "sse41.dpps")
      return UpgradeX86IntrinsicsWith8BitMask(F, Intrinsic::x86_sse41_dpps,
                                              NewFn);
    if (IsX86 && Name == "sse41.mpsadbw")
      return UpgradeX86IntrinsicsWith8BitMask(F, Intrinsic::x86_sse41_mpsadbw,
                                              NewFn);
    if (IsX86 && Name == "avx.dp.ps.256")
      return UpgradeX86IntrinsicsWith8BitMask(F, Intrinsic::x86_avx_dp_ps_256,
                                              NewFn);
    if (IsX86 && Name == "avx2.mpsadbw")
      return UpgradeX86IntrinsicsWith8BitMask(F, Intrinsic::x86_avx2_mpsadbw,
                                              NewFn);

    // frcz.ss/sd may need to have an argument dropped
    if (IsX86 && Name.startswith("xop.vfrcz.ss") && F->arg_size() == 2) {
      F->setName(Name + ".old");
      NewFn = Intrinsic::getDeclaration(F->getParent(),
                                        Intrinsic::x86_xop_vfrcz_ss);
      return true;
    }
    if (IsX86 && Name.startswith("xop.vfrcz.sd") && F->arg_size() == 2) {
      F->setName(Name + ".old");
      NewFn = Intrinsic::getDeclaration(F->getParent(),
                                        Intrinsic::x86_xop_vfrcz_sd);
      return true;
    }
    if (IsX86 && (Name.startswith("avx512.mask.pslli.") ||
                  Name.startswith("avx512.mask.psrai.") ||
                  Name.startswith("avx512.mask.psrli."))) {
      Intrinsic::ID ShiftID;
      if (Name.slice(12, 16) == "psll")
        ShiftID = Name[18] == 'd' ? Intrinsic::x86_avx512_mask_psll_di_512
                                  : Intrinsic::x86_avx512_mask_psll_qi_512;
      else if (Name.slice(12, 16) == "psra")
        ShiftID = Name[18] == 'd' ? Intrinsic::x86_avx512_mask_psra_di_512
                                  : Intrinsic::x86_avx512_mask_psra_qi_512;
      else
        ShiftID = Name[18] == 'd' ? Intrinsic::x86_avx512_mask_psrl_di_512
                                  : Intrinsic::x86_avx512_mask_psrl_qi_512;
      F->setName("llvm.x86." + Name + ".old");
      NewFn = Intrinsic::getDeclaration(F->getParent(), ShiftID);
      return true;
    }
    // Fix the FMA4 intrinsics to remove the 4
    if (IsX86 && Name.startswith("fma4.")) {
      F->setName("llvm.x86.fma" + Name.substr(5));
      NewFn = F;
      return true;
    }
    // Upgrade any XOP PERMIL2 index operand still using a float/double vector.
    if (IsX86 && Name.startswith("xop.vpermil2")) {
      auto Params = F->getFunctionType()->params();
      auto Idx = Params[2];
      if (Idx->getScalarType()->isFloatingPointTy()) {
        F->setName("llvm.x86." + Name + ".old");
        unsigned IdxSize = Idx->getPrimitiveSizeInBits();
        unsigned EltSize = Idx->getScalarSizeInBits();
        Intrinsic::ID Permil2ID;
        if (EltSize == 64 && IdxSize == 128)
          Permil2ID = Intrinsic::x86_xop_vpermil2pd;
        else if (EltSize == 32 && IdxSize == 128)
          Permil2ID = Intrinsic::x86_xop_vpermil2ps;
        else if (EltSize == 64 && IdxSize == 256)
          Permil2ID = Intrinsic::x86_xop_vpermil2pd_256;
        else
          Permil2ID = Intrinsic::x86_xop_vpermil2ps_256;
        NewFn = Intrinsic::getDeclaration(F->getParent(), Permil2ID);
        return true;
      }
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
  NewFn = nullptr;
  bool Upgraded = UpgradeIntrinsicFunction1(F, NewFn);
  assert(F != NewFn && "Intrinsic function upgraded to the same function");

  // Upgrade intrinsic attributes.  This does not change the function.
  if (NewFn)
    F = NewFn;
  if (Intrinsic::ID id = F->getIntrinsicID())
    F->setAttributes(Intrinsic::getAttributes(F->getContext(), id));
  return Upgraded;
}

bool llvm::UpgradeGlobalVariable(GlobalVariable *GV) {
  // Nothing to do yet.
  return false;
}

// Handles upgrading SSE2/AVX2/AVX512BW PSLLDQ intrinsics by converting them
// to byte shuffles.
static Value *UpgradeX86PSLLDQIntrinsics(IRBuilder<> &Builder,
                                         Value *Op, unsigned Shift) {
  Type *ResultTy = Op->getType();
  unsigned NumElts = ResultTy->getVectorNumElements() * 8;

  // Bitcast from a 64-bit element type to a byte element type.
  Type *VecTy = VectorType::get(Builder.getInt8Ty(), NumElts);
  Op = Builder.CreateBitCast(Op, VecTy, "cast");

  // We'll be shuffling in zeroes.
  Value *Res = Constant::getNullValue(VecTy);

  // If shift is less than 16, emit a shuffle to move the bytes. Otherwise,
  // we'll just return the zero vector.
  if (Shift < 16) {
    uint32_t Idxs[64];
    // 256/512-bit version is split into 2/4 16-byte lanes.
    for (unsigned l = 0; l != NumElts; l += 16)
      for (unsigned i = 0; i != 16; ++i) {
        unsigned Idx = NumElts + i - Shift;
        if (Idx < NumElts)
          Idx -= NumElts - 16; // end of lane, switch operand.
        Idxs[l + i] = Idx + l;
      }

    Res = Builder.CreateShuffleVector(Res, Op, makeArrayRef(Idxs, NumElts));
  }

  // Bitcast back to a 64-bit element type.
  return Builder.CreateBitCast(Res, ResultTy, "cast");
}

// Handles upgrading SSE2/AVX2/AVX512BW PSRLDQ intrinsics by converting them
// to byte shuffles.
static Value *UpgradeX86PSRLDQIntrinsics(IRBuilder<> &Builder, Value *Op,
                                         unsigned Shift) {
  Type *ResultTy = Op->getType();
  unsigned NumElts = ResultTy->getVectorNumElements() * 8;

  // Bitcast from a 64-bit element type to a byte element type.
  Type *VecTy = VectorType::get(Builder.getInt8Ty(), NumElts);
  Op = Builder.CreateBitCast(Op, VecTy, "cast");

  // We'll be shuffling in zeroes.
  Value *Res = Constant::getNullValue(VecTy);

  // If shift is less than 16, emit a shuffle to move the bytes. Otherwise,
  // we'll just return the zero vector.
  if (Shift < 16) {
    uint32_t Idxs[64];
    // 256/512-bit version is split into 2/4 16-byte lanes.
    for (unsigned l = 0; l != NumElts; l += 16)
      for (unsigned i = 0; i != 16; ++i) {
        unsigned Idx = i + Shift;
        if (Idx >= 16)
          Idx += NumElts - 16; // end of lane, switch operand.
        Idxs[l + i] = Idx + l;
      }

    Res = Builder.CreateShuffleVector(Op, Res, makeArrayRef(Idxs, NumElts));
  }

  // Bitcast back to a 64-bit element type.
  return Builder.CreateBitCast(Res, ResultTy, "cast");
}

static Value *getX86MaskVec(IRBuilder<> &Builder, Value *Mask,
                            unsigned NumElts) {
  llvm::VectorType *MaskTy = llvm::VectorType::get(Builder.getInt1Ty(),
                             cast<IntegerType>(Mask->getType())->getBitWidth());
  Mask = Builder.CreateBitCast(Mask, MaskTy);

  // If we have less than 8 elements, then the starting mask was an i8 and
  // we need to extract down to the right number of elements.
  if (NumElts < 8) {
    uint32_t Indices[4];
    for (unsigned i = 0; i != NumElts; ++i)
      Indices[i] = i;
    Mask = Builder.CreateShuffleVector(Mask, Mask,
                                       makeArrayRef(Indices, NumElts),
                                       "extract");
  }

  return Mask;
}

static Value *EmitX86Select(IRBuilder<> &Builder, Value *Mask,
                            Value *Op0, Value *Op1) {
  // If the mask is all ones just emit the align operation.
  if (const auto *C = dyn_cast<Constant>(Mask))
    if (C->isAllOnesValue())
      return Op0;

  Mask = getX86MaskVec(Builder, Mask, Op0->getType()->getVectorNumElements());
  return Builder.CreateSelect(Mask, Op0, Op1);
}

static Value *UpgradeX86PALIGNRIntrinsics(IRBuilder<> &Builder,
                                          Value *Op0, Value *Op1, Value *Shift,
                                          Value *Passthru, Value *Mask) {
  unsigned ShiftVal = cast<llvm::ConstantInt>(Shift)->getZExtValue();

  unsigned NumElts = Op0->getType()->getVectorNumElements();
  assert(NumElts % 16 == 0);

  // If palignr is shifting the pair of vectors more than the size of two
  // lanes, emit zero.
  if (ShiftVal >= 32)
    return llvm::Constant::getNullValue(Op0->getType());

  // If palignr is shifting the pair of input vectors more than one lane,
  // but less than two lanes, convert to shifting in zeroes.
  if (ShiftVal > 16) {
    ShiftVal -= 16;
    Op1 = Op0;
    Op0 = llvm::Constant::getNullValue(Op0->getType());
  }

  uint32_t Indices[64];
  // 256-bit palignr operates on 128-bit lanes so we need to handle that
  for (unsigned l = 0; l != NumElts; l += 16) {
    for (unsigned i = 0; i != 16; ++i) {
      unsigned Idx = ShiftVal + i;
      if (Idx >= 16)
        Idx += NumElts - 16; // End of lane, switch operand.
      Indices[l + i] = Idx + l;
    }
  }

  Value *Align = Builder.CreateShuffleVector(Op1, Op0,
                                             makeArrayRef(Indices, NumElts),
                                             "palignr");

  return EmitX86Select(Builder, Mask, Align, Passthru);
}

static Value *UpgradeMaskedStore(IRBuilder<> &Builder,
                                 Value *Ptr, Value *Data, Value *Mask,
                                 bool Aligned) {
  // Cast the pointer to the right type.
  Ptr = Builder.CreateBitCast(Ptr,
                              llvm::PointerType::getUnqual(Data->getType()));
  unsigned Align =
    Aligned ? cast<VectorType>(Data->getType())->getBitWidth() / 8 : 1;

  // If the mask is all ones just emit a regular store.
  if (const auto *C = dyn_cast<Constant>(Mask))
    if (C->isAllOnesValue())
      return Builder.CreateAlignedStore(Data, Ptr, Align);

  // Convert the mask from an integer type to a vector of i1.
  unsigned NumElts = Data->getType()->getVectorNumElements();
  Mask = getX86MaskVec(Builder, Mask, NumElts);
  return Builder.CreateMaskedStore(Data, Ptr, Align, Mask);
}

static Value *UpgradeMaskedLoad(IRBuilder<> &Builder,
                                Value *Ptr, Value *Passthru, Value *Mask,
                                bool Aligned) {
  // Cast the pointer to the right type.
  Ptr = Builder.CreateBitCast(Ptr,
                             llvm::PointerType::getUnqual(Passthru->getType()));
  unsigned Align =
    Aligned ? cast<VectorType>(Passthru->getType())->getBitWidth() / 8 : 1;

  // If the mask is all ones just emit a regular store.
  if (const auto *C = dyn_cast<Constant>(Mask))
    if (C->isAllOnesValue())
      return Builder.CreateAlignedLoad(Ptr, Align);

  // Convert the mask from an integer type to a vector of i1.
  unsigned NumElts = Passthru->getType()->getVectorNumElements();
  Mask = getX86MaskVec(Builder, Mask, NumElts);
  return Builder.CreateMaskedLoad(Ptr, Align, Mask, Passthru);
}

static Value *upgradeIntMinMax(IRBuilder<> &Builder, CallInst &CI,
                               ICmpInst::Predicate Pred) {
  Value *Op0 = CI.getArgOperand(0);
  Value *Op1 = CI.getArgOperand(1);
  Value *Cmp = Builder.CreateICmp(Pred, Op0, Op1);
  return Builder.CreateSelect(Cmp, Op0, Op1);
}

static Value *upgradeMaskedCompare(IRBuilder<> &Builder, CallInst &CI,
                                   ICmpInst::Predicate Pred) {
  Value *Op0 = CI.getArgOperand(0);
  unsigned NumElts = Op0->getType()->getVectorNumElements();
  Value *Cmp = Builder.CreateICmp(Pred, Op0, CI.getArgOperand(1));

  Value *Mask = CI.getArgOperand(2);
  const auto *C = dyn_cast<Constant>(Mask);
  if (!C || !C->isAllOnesValue())
    Cmp = Builder.CreateAnd(Cmp, getX86MaskVec(Builder, Mask, NumElts));

  if (NumElts < 8) {
    uint32_t Indices[8];
    for (unsigned i = 0; i != NumElts; ++i)
      Indices[i] = i;
    for (unsigned i = NumElts; i != 8; ++i)
      Indices[i] = NumElts + i % NumElts;
    Cmp = Builder.CreateShuffleVector(Cmp,
                                      Constant::getNullValue(Cmp->getType()),
                                      Indices);
  }
  return Builder.CreateBitCast(Cmp, IntegerType::get(CI.getContext(),
                                                     std::max(NumElts, 8U)));
}

/// Upgrade a call to an old intrinsic. All argument and return casting must be
/// provided to seamlessly integrate with existing context.
void llvm::UpgradeIntrinsicCall(CallInst *CI, Function *NewFn) {
  Function *F = CI->getCalledFunction();
  LLVMContext &C = CI->getContext();
  IRBuilder<> Builder(C);
  Builder.SetInsertPoint(CI->getParent(), CI->getIterator());

  assert(F && "Intrinsic call is not direct?");

  if (!NewFn) {
    // Get the Function's name.
    StringRef Name = F->getName();

    assert(Name.startswith("llvm.") && "Intrinsic doesn't start with 'llvm.'");
    Name = Name.substr(5);

    bool IsX86 = Name.startswith("x86.");
    if (IsX86)
      Name = Name.substr(4);

    Value *Rep;
    // Upgrade packed integer vector compare intrinsics to compare instructions.
    if (IsX86 && (Name.startswith("sse2.pcmpeq.") ||
                  Name.startswith("avx2.pcmpeq."))) {
      Rep = Builder.CreateICmpEQ(CI->getArgOperand(0), CI->getArgOperand(1),
                                 "pcmpeq");
      Rep = Builder.CreateSExt(Rep, CI->getType(), "");
    } else if (IsX86 && (Name.startswith("sse2.pcmpgt.") ||
                         Name.startswith("avx2.pcmpgt."))) {
      Rep = Builder.CreateICmpSGT(CI->getArgOperand(0), CI->getArgOperand(1),
                                  "pcmpgt");
      Rep = Builder.CreateSExt(Rep, CI->getType(), "");
    } else if (IsX86 && Name.startswith("avx512.mask.pcmpeq.")) {
      Rep = upgradeMaskedCompare(Builder, *CI, ICmpInst::ICMP_EQ);
    } else if (IsX86 && Name.startswith("avx512.mask.pcmpgt.")) {
      Rep = upgradeMaskedCompare(Builder, *CI, ICmpInst::ICMP_SGT);
    } else if (IsX86 && (Name == "sse41.pmaxsb" ||
                         Name == "sse2.pmaxs.w" ||
                         Name == "sse41.pmaxsd" ||
                         Name.startswith("avx2.pmaxs"))) {
      Rep = upgradeIntMinMax(Builder, *CI, ICmpInst::ICMP_SGT);
    } else if (IsX86 && (Name == "sse2.pmaxu.b" ||
                         Name == "sse41.pmaxuw" ||
                         Name == "sse41.pmaxud" ||
                         Name.startswith("avx2.pmaxu"))) {
      Rep = upgradeIntMinMax(Builder, *CI, ICmpInst::ICMP_UGT);
    } else if (IsX86 && (Name == "sse41.pminsb" ||
                         Name == "sse2.pmins.w" ||
                         Name == "sse41.pminsd" ||
                         Name.startswith("avx2.pmins"))) {
      Rep = upgradeIntMinMax(Builder, *CI, ICmpInst::ICMP_SLT);
    } else if (IsX86 && (Name == "sse2.pminu.b" ||
                         Name == "sse41.pminuw" ||
                         Name == "sse41.pminud" ||
                         Name.startswith("avx2.pminu"))) {
      Rep = upgradeIntMinMax(Builder, *CI, ICmpInst::ICMP_ULT);
    } else if (IsX86 && (Name == "sse2.cvtdq2pd" ||
                         Name == "sse2.cvtps2pd" ||
                         Name == "avx.cvtdq2.pd.256" ||
                         Name == "avx.cvt.ps2.pd.256")) {
      // Lossless i32/float to double conversion.
      // Extract the bottom elements if necessary and convert to double vector.
      Value *Src = CI->getArgOperand(0);
      VectorType *SrcTy = cast<VectorType>(Src->getType());
      VectorType *DstTy = cast<VectorType>(CI->getType());
      Rep = CI->getArgOperand(0);

      unsigned NumDstElts = DstTy->getNumElements();
      if (NumDstElts < SrcTy->getNumElements()) {
        assert(NumDstElts == 2 && "Unexpected vector size");
        uint32_t ShuffleMask[2] = { 0, 1 };
        Rep = Builder.CreateShuffleVector(Rep, UndefValue::get(SrcTy),
                                          ShuffleMask);
      }

      bool Int2Double = (StringRef::npos != Name.find("cvtdq2"));
      if (Int2Double)
        Rep = Builder.CreateSIToFP(Rep, DstTy, "cvtdq2pd");
      else
        Rep = Builder.CreateFPExt(Rep, DstTy, "cvtps2pd");
    } else if (IsX86 && Name.startswith("sse4a.movnt.")) {
      Module *M = F->getParent();
      SmallVector<Metadata *, 1> Elts;
      Elts.push_back(
          ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(C), 1)));
      MDNode *Node = MDNode::get(C, Elts);

      Value *Arg0 = CI->getArgOperand(0);
      Value *Arg1 = CI->getArgOperand(1);

      // Nontemporal (unaligned) store of the 0'th element of the float/double
      // vector.
      Type *SrcEltTy = cast<VectorType>(Arg1->getType())->getElementType();
      PointerType *EltPtrTy = PointerType::getUnqual(SrcEltTy);
      Value *Addr = Builder.CreateBitCast(Arg0, EltPtrTy, "cast");
      Value *Extract =
          Builder.CreateExtractElement(Arg1, (uint64_t)0, "extractelement");

      StoreInst *SI = Builder.CreateAlignedStore(Extract, Addr, 1);
      SI->setMetadata(M->getMDKindID("nontemporal"), Node);

      // Remove intrinsic.
      CI->eraseFromParent();
      return;
    } else if (IsX86 && (Name.startswith("avx.movnt.") ||
                         Name.startswith("avx512.storent."))) {
      Module *M = F->getParent();
      SmallVector<Metadata *, 1> Elts;
      Elts.push_back(
          ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(C), 1)));
      MDNode *Node = MDNode::get(C, Elts);

      Value *Arg0 = CI->getArgOperand(0);
      Value *Arg1 = CI->getArgOperand(1);

      // Convert the type of the pointer to a pointer to the stored type.
      Value *BC = Builder.CreateBitCast(Arg0,
                                        PointerType::getUnqual(Arg1->getType()),
                                        "cast");
      VectorType *VTy = cast<VectorType>(Arg1->getType());
      StoreInst *SI = Builder.CreateAlignedStore(Arg1, BC,
                                                 VTy->getBitWidth() / 8);
      SI->setMetadata(M->getMDKindID("nontemporal"), Node);

      // Remove intrinsic.
      CI->eraseFromParent();
      return;
    } else if (IsX86 && Name == "sse2.storel.dq") {
      Value *Arg0 = CI->getArgOperand(0);
      Value *Arg1 = CI->getArgOperand(1);

      Type *NewVecTy = VectorType::get(Type::getInt64Ty(C), 2);
      Value *BC0 = Builder.CreateBitCast(Arg1, NewVecTy, "cast");
      Value *Elt = Builder.CreateExtractElement(BC0, (uint64_t)0);
      Value *BC = Builder.CreateBitCast(Arg0,
                                        PointerType::getUnqual(Elt->getType()),
                                        "cast");
      Builder.CreateAlignedStore(Elt, BC, 1);

      // Remove intrinsic.
      CI->eraseFromParent();
      return;
    } else if (IsX86 && (Name.startswith("sse.storeu.") ||
                         Name.startswith("sse2.storeu.") ||
                         Name.startswith("avx.storeu."))) {
      Value *Arg0 = CI->getArgOperand(0);
      Value *Arg1 = CI->getArgOperand(1);

      Arg0 = Builder.CreateBitCast(Arg0,
                                   PointerType::getUnqual(Arg1->getType()),
                                   "cast");
      Builder.CreateAlignedStore(Arg1, Arg0, 1);

      // Remove intrinsic.
      CI->eraseFromParent();
      return;
    } else if (IsX86 && (Name.startswith("avx512.mask.storeu.p") ||
                         Name.startswith("avx512.mask.storeu.b.") ||
                         Name.startswith("avx512.mask.storeu.w.") ||
                         Name.startswith("avx512.mask.storeu.d.") ||
                         Name.startswith("avx512.mask.storeu.q."))) {
      UpgradeMaskedStore(Builder, CI->getArgOperand(0), CI->getArgOperand(1),
                         CI->getArgOperand(2), /*Aligned*/false);

      // Remove intrinsic.
      CI->eraseFromParent();
      return;
    } else if (IsX86 && (Name.startswith("avx512.mask.store.p") ||
                         Name.startswith("avx512.mask.store.b.") ||
                         Name.startswith("avx512.mask.store.w.") ||
                         Name.startswith("avx512.mask.store.d.") ||
                         Name.startswith("avx512.mask.store.q."))) {
      UpgradeMaskedStore(Builder, CI->getArgOperand(0), CI->getArgOperand(1),
                         CI->getArgOperand(2), /*Aligned*/true);

      // Remove intrinsic.
      CI->eraseFromParent();
      return;
    } else if (IsX86 && (Name.startswith("avx512.mask.loadu.p") ||
                         Name.startswith("avx512.mask.loadu.b.") ||
                         Name.startswith("avx512.mask.loadu.w.") ||
                         Name.startswith("avx512.mask.loadu.d.") ||
                         Name.startswith("avx512.mask.loadu.q."))) {
      Rep = UpgradeMaskedLoad(Builder, CI->getArgOperand(0),
                              CI->getArgOperand(1), CI->getArgOperand(2),
                              /*Aligned*/false);
    } else if (IsX86 && (Name.startswith("avx512.mask.load.p") ||
                         Name.startswith("avx512.mask.load.b.") ||
                         Name.startswith("avx512.mask.load.w.") ||
                         Name.startswith("avx512.mask.load.d.") ||
                         Name.startswith("avx512.mask.load.q."))) {
      Rep = UpgradeMaskedLoad(Builder, CI->getArgOperand(0),
                              CI->getArgOperand(1),CI->getArgOperand(2),
                              /*Aligned*/true);
    } else if (IsX86 && Name.startswith("xop.vpcom")) {
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

      Name = Name.substr(9); // strip off "xop.vpcom"
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
      else if (Name.startswith("false"))
        Imm = 6;
      else if (Name.startswith("true"))
        Imm = 7;
      else
        llvm_unreachable("Unknown condition");

      Function *VPCOM = Intrinsic::getDeclaration(F->getParent(), intID);
      Rep =
          Builder.CreateCall(VPCOM, {CI->getArgOperand(0), CI->getArgOperand(1),
                                     Builder.getInt8(Imm)});
    } else if (IsX86 && Name == "xop.vpcmov") {
      Value *Arg0 = CI->getArgOperand(0);
      Value *Arg1 = CI->getArgOperand(1);
      Value *Sel = CI->getArgOperand(2);
      unsigned NumElts = CI->getType()->getVectorNumElements();
      Constant *MinusOne = ConstantVector::getSplat(NumElts, Builder.getInt64(-1));
      Value *NotSel = Builder.CreateXor(Sel, MinusOne);
      Value *Sel0 = Builder.CreateAnd(Arg0, Sel);
      Value *Sel1 = Builder.CreateAnd(Arg1, NotSel);
      Rep = Builder.CreateOr(Sel0, Sel1);
    } else if (IsX86 && Name == "sse42.crc32.64.8") {
      Function *CRC32 = Intrinsic::getDeclaration(F->getParent(),
                                               Intrinsic::x86_sse42_crc32_32_8);
      Value *Trunc0 = Builder.CreateTrunc(CI->getArgOperand(0), Type::getInt32Ty(C));
      Rep = Builder.CreateCall(CRC32, {Trunc0, CI->getArgOperand(1)});
      Rep = Builder.CreateZExt(Rep, CI->getType(), "");
    } else if (IsX86 && Name.startswith("avx.vbroadcast")) {
      // Replace broadcasts with a series of insertelements.
      Type *VecTy = CI->getType();
      Type *EltTy = VecTy->getVectorElementType();
      unsigned EltNum = VecTy->getVectorNumElements();
      Value *Cast = Builder.CreateBitCast(CI->getArgOperand(0),
                                          EltTy->getPointerTo());
      Value *Load = Builder.CreateLoad(EltTy, Cast);
      Type *I32Ty = Type::getInt32Ty(C);
      Rep = UndefValue::get(VecTy);
      for (unsigned I = 0; I < EltNum; ++I)
        Rep = Builder.CreateInsertElement(Rep, Load,
                                          ConstantInt::get(I32Ty, I));
    } else if (IsX86 && (Name.startswith("sse41.pmovsx") ||
                         Name.startswith("sse41.pmovzx") ||
                         Name.startswith("avx2.pmovsx") ||
                         Name.startswith("avx2.pmovzx"))) {
      VectorType *SrcTy = cast<VectorType>(CI->getArgOperand(0)->getType());
      VectorType *DstTy = cast<VectorType>(CI->getType());
      unsigned NumDstElts = DstTy->getNumElements();

      // Extract a subvector of the first NumDstElts lanes and sign/zero extend.
      SmallVector<uint32_t, 8> ShuffleMask(NumDstElts);
      for (unsigned i = 0; i != NumDstElts; ++i)
        ShuffleMask[i] = i;

      Value *SV = Builder.CreateShuffleVector(
          CI->getArgOperand(0), UndefValue::get(SrcTy), ShuffleMask);

      bool DoSext = (StringRef::npos != Name.find("pmovsx"));
      Rep = DoSext ? Builder.CreateSExt(SV, DstTy)
                   : Builder.CreateZExt(SV, DstTy);
    } else if (IsX86 && Name == "avx2.vbroadcasti128") {
      // Replace vbroadcasts with a vector shuffle.
      Type *VT = VectorType::get(Type::getInt64Ty(C), 2);
      Value *Op = Builder.CreatePointerCast(CI->getArgOperand(0),
                                            PointerType::getUnqual(VT));
      Value *Load = Builder.CreateLoad(VT, Op);
      uint32_t Idxs[4] = { 0, 1, 0, 1 };
      Rep = Builder.CreateShuffleVector(Load, UndefValue::get(Load->getType()),
                                        Idxs);
    } else if (IsX86 && (Name.startswith("avx2.pbroadcast") ||
                         Name.startswith("avx2.vbroadcast") ||
                         Name.startswith("avx512.pbroadcast") ||
                         Name.startswith("avx512.mask.broadcast.s"))) {
      // Replace vp?broadcasts with a vector shuffle.
      Value *Op = CI->getArgOperand(0);
      unsigned NumElts = CI->getType()->getVectorNumElements();
      Type *MaskTy = VectorType::get(Type::getInt32Ty(C), NumElts);
      Rep = Builder.CreateShuffleVector(Op, UndefValue::get(Op->getType()),
                                        Constant::getNullValue(MaskTy));

      if (CI->getNumArgOperands() == 3)
        Rep = EmitX86Select(Builder, CI->getArgOperand(2), Rep,
                            CI->getArgOperand(1));
    } else if (IsX86 && Name.startswith("avx512.mask.palignr.")) {
      Rep = UpgradeX86PALIGNRIntrinsics(Builder, CI->getArgOperand(0),
                                        CI->getArgOperand(1),
                                        CI->getArgOperand(2),
                                        CI->getArgOperand(3),
                                        CI->getArgOperand(4));
    } else if (IsX86 && (Name == "sse2.psll.dq" ||
                         Name == "avx2.psll.dq")) {
      // 128/256-bit shift left specified in bits.
      unsigned Shift = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      Rep = UpgradeX86PSLLDQIntrinsics(Builder, CI->getArgOperand(0),
                                       Shift / 8); // Shift is in bits.
    } else if (IsX86 && (Name == "sse2.psrl.dq" ||
                         Name == "avx2.psrl.dq")) {
      // 128/256-bit shift right specified in bits.
      unsigned Shift = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      Rep = UpgradeX86PSRLDQIntrinsics(Builder, CI->getArgOperand(0),
                                       Shift / 8); // Shift is in bits.
    } else if (IsX86 && (Name == "sse2.psll.dq.bs" ||
                         Name == "avx2.psll.dq.bs" ||
                         Name == "avx512.psll.dq.512")) {
      // 128/256/512-bit shift left specified in bytes.
      unsigned Shift = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      Rep = UpgradeX86PSLLDQIntrinsics(Builder, CI->getArgOperand(0), Shift);
    } else if (IsX86 && (Name == "sse2.psrl.dq.bs" ||
                         Name == "avx2.psrl.dq.bs" ||
                         Name == "avx512.psrl.dq.512")) {
      // 128/256/512-bit shift right specified in bytes.
      unsigned Shift = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      Rep = UpgradeX86PSRLDQIntrinsics(Builder, CI->getArgOperand(0), Shift);
    } else if (IsX86 && (Name == "sse41.pblendw" ||
                         Name.startswith("sse41.blendp") ||
                         Name.startswith("avx.blend.p") ||
                         Name == "avx2.pblendw" ||
                         Name.startswith("avx2.pblendd."))) {
      Value *Op0 = CI->getArgOperand(0);
      Value *Op1 = CI->getArgOperand(1);
      unsigned Imm = cast <ConstantInt>(CI->getArgOperand(2))->getZExtValue();
      VectorType *VecTy = cast<VectorType>(CI->getType());
      unsigned NumElts = VecTy->getNumElements();

      SmallVector<uint32_t, 16> Idxs(NumElts);
      for (unsigned i = 0; i != NumElts; ++i)
        Idxs[i] = ((Imm >> (i%8)) & 1) ? i + NumElts : i;

      Rep = Builder.CreateShuffleVector(Op0, Op1, Idxs);
    } else if (IsX86 && (Name.startswith("avx.vinsertf128.") ||
                         Name == "avx2.vinserti128")) {
      Value *Op0 = CI->getArgOperand(0);
      Value *Op1 = CI->getArgOperand(1);
      unsigned Imm = cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue();
      VectorType *VecTy = cast<VectorType>(CI->getType());
      unsigned NumElts = VecTy->getNumElements();

      // Mask off the high bits of the immediate value; hardware ignores those.
      Imm = Imm & 1;

      // Extend the second operand into a vector that is twice as big.
      Value *UndefV = UndefValue::get(Op1->getType());
      SmallVector<uint32_t, 8> Idxs(NumElts);
      for (unsigned i = 0; i != NumElts; ++i)
        Idxs[i] = i;
      Rep = Builder.CreateShuffleVector(Op1, UndefV, Idxs);

      // Insert the second operand into the first operand.

      // Note that there is no guarantee that instruction lowering will actually
      // produce a vinsertf128 instruction for the created shuffles. In
      // particular, the 0 immediate case involves no lane changes, so it can
      // be handled as a blend.

      // Example of shuffle mask for 32-bit elements:
      // Imm = 1  <i32 0, i32 1, i32 2,  i32 3,  i32 8, i32 9, i32 10, i32 11>
      // Imm = 0  <i32 8, i32 9, i32 10, i32 11, i32 4, i32 5, i32 6,  i32 7 >

      // The low half of the result is either the low half of the 1st operand
      // or the low half of the 2nd operand (the inserted vector).
      for (unsigned i = 0; i != NumElts / 2; ++i)
        Idxs[i] = Imm ? i : (i + NumElts);
      // The high half of the result is either the low half of the 2nd operand
      // (the inserted vector) or the high half of the 1st operand.
      for (unsigned i = NumElts / 2; i != NumElts; ++i)
        Idxs[i] = Imm ? (i + NumElts / 2) : i;
      Rep = Builder.CreateShuffleVector(Op0, Rep, Idxs);
    } else if (IsX86 && (Name.startswith("avx.vextractf128.") ||
                         Name == "avx2.vextracti128")) {
      Value *Op0 = CI->getArgOperand(0);
      unsigned Imm = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      VectorType *VecTy = cast<VectorType>(CI->getType());
      unsigned NumElts = VecTy->getNumElements();

      // Mask off the high bits of the immediate value; hardware ignores those.
      Imm = Imm & 1;

      // Get indexes for either the high half or low half of the input vector.
      SmallVector<uint32_t, 4> Idxs(NumElts);
      for (unsigned i = 0; i != NumElts; ++i) {
        Idxs[i] = Imm ? (i + NumElts) : i;
      }

      Value *UndefV = UndefValue::get(Op0->getType());
      Rep = Builder.CreateShuffleVector(Op0, UndefV, Idxs);
    } else if (!IsX86 && Name == "stackprotectorcheck") {
      Rep = nullptr;
    } else if (IsX86 && (Name.startswith("avx512.mask.perm.df.") ||
                         Name.startswith("avx512.mask.perm.di."))) {
      Value *Op0 = CI->getArgOperand(0);
      unsigned Imm = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      VectorType *VecTy = cast<VectorType>(CI->getType());
      unsigned NumElts = VecTy->getNumElements();

      SmallVector<uint32_t, 8> Idxs(NumElts);
      for (unsigned i = 0; i != NumElts; ++i)
        Idxs[i] = (i & ~0x3) + ((Imm >> (2 * (i & 0x3))) & 3);

      Rep = Builder.CreateShuffleVector(Op0, Op0, Idxs);

      if (CI->getNumArgOperands() == 4)
        Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                            CI->getArgOperand(2));
    } else if (IsX86 && (Name.startswith("avx.vpermil.") ||
                         Name == "sse2.pshuf.d" ||
                         Name.startswith("avx512.mask.vpermil.p") ||
                         Name.startswith("avx512.mask.pshuf.d."))) {
      Value *Op0 = CI->getArgOperand(0);
      unsigned Imm = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      VectorType *VecTy = cast<VectorType>(CI->getType());
      unsigned NumElts = VecTy->getNumElements();
      // Calculate the size of each index in the immediate.
      unsigned IdxSize = 64 / VecTy->getScalarSizeInBits();
      unsigned IdxMask = ((1 << IdxSize) - 1);

      SmallVector<uint32_t, 8> Idxs(NumElts);
      // Lookup the bits for this element, wrapping around the immediate every
      // 8-bits. Elements are grouped into sets of 2 or 4 elements so we need
      // to offset by the first index of each group.
      for (unsigned i = 0; i != NumElts; ++i)
        Idxs[i] = ((Imm >> ((i * IdxSize) % 8)) & IdxMask) | (i & ~IdxMask);

      Rep = Builder.CreateShuffleVector(Op0, Op0, Idxs);

      if (CI->getNumArgOperands() == 4)
        Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                            CI->getArgOperand(2));
    } else if (IsX86 && (Name == "sse2.pshufl.w" ||
                         Name.startswith("avx512.mask.pshufl.w."))) {
      Value *Op0 = CI->getArgOperand(0);
      unsigned Imm = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      unsigned NumElts = CI->getType()->getVectorNumElements();

      SmallVector<uint32_t, 16> Idxs(NumElts);
      for (unsigned l = 0; l != NumElts; l += 8) {
        for (unsigned i = 0; i != 4; ++i)
          Idxs[i + l] = ((Imm >> (2 * i)) & 0x3) + l;
        for (unsigned i = 4; i != 8; ++i)
          Idxs[i + l] = i + l;
      }

      Rep = Builder.CreateShuffleVector(Op0, Op0, Idxs);

      if (CI->getNumArgOperands() == 4)
        Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                            CI->getArgOperand(2));
    } else if (IsX86 && (Name == "sse2.pshufh.w" ||
                         Name.startswith("avx512.mask.pshufh.w."))) {
      Value *Op0 = CI->getArgOperand(0);
      unsigned Imm = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      unsigned NumElts = CI->getType()->getVectorNumElements();

      SmallVector<uint32_t, 16> Idxs(NumElts);
      for (unsigned l = 0; l != NumElts; l += 8) {
        for (unsigned i = 0; i != 4; ++i)
          Idxs[i + l] = i + l;
        for (unsigned i = 0; i != 4; ++i)
          Idxs[i + l + 4] = ((Imm >> (2 * i)) & 0x3) + 4 + l;
      }

      Rep = Builder.CreateShuffleVector(Op0, Op0, Idxs);

      if (CI->getNumArgOperands() == 4)
        Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                            CI->getArgOperand(2));
    } else if (IsX86 && (Name.startswith("avx512.mask.movddup") ||
                         Name.startswith("avx512.mask.movshdup") ||
                         Name.startswith("avx512.mask.movsldup"))) {
      Value *Op0 = CI->getArgOperand(0);
      unsigned NumElts = CI->getType()->getVectorNumElements();
      unsigned NumLaneElts = 128/CI->getType()->getScalarSizeInBits();

      unsigned Offset = 0;
      if (Name.startswith("avx512.mask.movshdup."))
        Offset = 1;

      SmallVector<uint32_t, 16> Idxs(NumElts);
      for (unsigned l = 0; l != NumElts; l += NumLaneElts)
        for (unsigned i = 0; i != NumLaneElts; i += 2) {
          Idxs[i + l + 0] = i + l + Offset;
          Idxs[i + l + 1] = i + l + Offset;
        }

      Rep = Builder.CreateShuffleVector(Op0, Op0, Idxs);

      Rep = EmitX86Select(Builder, CI->getArgOperand(2), Rep,
                          CI->getArgOperand(1));
    } else if (IsX86 && (Name.startswith("avx512.mask.punpckl") ||
                         Name.startswith("avx512.mask.unpckl."))) {
      Value *Op0 = CI->getArgOperand(0);
      Value *Op1 = CI->getArgOperand(1);
      int NumElts = CI->getType()->getVectorNumElements();
      int NumLaneElts = 128/CI->getType()->getScalarSizeInBits();

      SmallVector<uint32_t, 64> Idxs(NumElts);
      for (int l = 0; l != NumElts; l += NumLaneElts)
        for (int i = 0; i != NumLaneElts; ++i)
          Idxs[i + l] = l + (i / 2) + NumElts * (i % 2);

      Rep = Builder.CreateShuffleVector(Op0, Op1, Idxs);

      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                          CI->getArgOperand(2));
    } else if (IsX86 && (Name.startswith("avx512.mask.punpckh") ||
                         Name.startswith("avx512.mask.unpckh."))) {
      Value *Op0 = CI->getArgOperand(0);
      Value *Op1 = CI->getArgOperand(1);
      int NumElts = CI->getType()->getVectorNumElements();
      int NumLaneElts = 128/CI->getType()->getScalarSizeInBits();

      SmallVector<uint32_t, 64> Idxs(NumElts);
      for (int l = 0; l != NumElts; l += NumLaneElts)
        for (int i = 0; i != NumLaneElts; ++i)
          Idxs[i + l] = (NumLaneElts / 2) + l + (i / 2) + NumElts * (i % 2);

      Rep = Builder.CreateShuffleVector(Op0, Op1, Idxs);

      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                          CI->getArgOperand(2));
    } else if (IsX86 && Name.startswith("avx512.mask.pand.")) {
      Rep = Builder.CreateAnd(CI->getArgOperand(0), CI->getArgOperand(1));
      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                          CI->getArgOperand(2));
    } else if (IsX86 && Name.startswith("avx512.mask.pandn.")) {
      Rep = Builder.CreateAnd(Builder.CreateNot(CI->getArgOperand(0)),
                              CI->getArgOperand(1));
      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                          CI->getArgOperand(2));
    } else if (IsX86 && Name.startswith("avx512.mask.por.")) {
      Rep = Builder.CreateOr(CI->getArgOperand(0), CI->getArgOperand(1));
      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                          CI->getArgOperand(2));
    } else if (IsX86 && Name.startswith("avx512.mask.pxor.")) {
      Rep = Builder.CreateXor(CI->getArgOperand(0), CI->getArgOperand(1));
      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                          CI->getArgOperand(2));
    } else {
      llvm_unreachable("Unknown function for CallInst upgrade.");
    }

    if (Rep)
      CI->replaceAllUsesWith(Rep);
    CI->eraseFromParent();
    return;
  }

  std::string Name = CI->getName();
  if (!Name.empty())
    CI->setName(Name + ".old");

  switch (NewFn->getIntrinsicID()) {
  default:
    llvm_unreachable("Unknown function for CallInst upgrade.");

  case Intrinsic::x86_avx512_mask_psll_di_512:
  case Intrinsic::x86_avx512_mask_psra_di_512:
  case Intrinsic::x86_avx512_mask_psrl_di_512:
  case Intrinsic::x86_avx512_mask_psll_qi_512:
  case Intrinsic::x86_avx512_mask_psra_qi_512:
  case Intrinsic::x86_avx512_mask_psrl_qi_512:
  case Intrinsic::arm_neon_vld1:
  case Intrinsic::arm_neon_vld2:
  case Intrinsic::arm_neon_vld3:
  case Intrinsic::arm_neon_vld4:
  case Intrinsic::arm_neon_vld2lane:
  case Intrinsic::arm_neon_vld3lane:
  case Intrinsic::arm_neon_vld4lane:
  case Intrinsic::arm_neon_vst1:
  case Intrinsic::arm_neon_vst2:
  case Intrinsic::arm_neon_vst3:
  case Intrinsic::arm_neon_vst4:
  case Intrinsic::arm_neon_vst2lane:
  case Intrinsic::arm_neon_vst3lane:
  case Intrinsic::arm_neon_vst4lane: {
    SmallVector<Value *, 4> Args(CI->arg_operands().begin(),
                                 CI->arg_operands().end());
    CI->replaceAllUsesWith(Builder.CreateCall(NewFn, Args));
    CI->eraseFromParent();
    return;
  }

  case Intrinsic::ctlz:
  case Intrinsic::cttz:
    assert(CI->getNumArgOperands() == 1 &&
           "Mismatch between function args and call args");
    CI->replaceAllUsesWith(Builder.CreateCall(
        NewFn, {CI->getArgOperand(0), Builder.getFalse()}, Name));
    CI->eraseFromParent();
    return;

  case Intrinsic::objectsize:
    CI->replaceAllUsesWith(Builder.CreateCall(
        NewFn, {CI->getArgOperand(0), CI->getArgOperand(1)}, Name));
    CI->eraseFromParent();
    return;

  case Intrinsic::ctpop: {
    CI->replaceAllUsesWith(Builder.CreateCall(NewFn, {CI->getArgOperand(0)}));
    CI->eraseFromParent();
    return;
  }

  case Intrinsic::x86_xop_vfrcz_ss:
  case Intrinsic::x86_xop_vfrcz_sd:
    CI->replaceAllUsesWith(
        Builder.CreateCall(NewFn, {CI->getArgOperand(1)}, Name));
    CI->eraseFromParent();
    return;

  case Intrinsic::x86_xop_vpermil2pd:
  case Intrinsic::x86_xop_vpermil2ps:
  case Intrinsic::x86_xop_vpermil2pd_256:
  case Intrinsic::x86_xop_vpermil2ps_256: {
    SmallVector<Value *, 4> Args(CI->arg_operands().begin(),
                                 CI->arg_operands().end());
    VectorType *FltIdxTy = cast<VectorType>(Args[2]->getType());
    VectorType *IntIdxTy = VectorType::getInteger(FltIdxTy);
    Args[2] = Builder.CreateBitCast(Args[2], IntIdxTy);
    CI->replaceAllUsesWith(Builder.CreateCall(NewFn, Args, Name));
    CI->eraseFromParent();
    return;
  }

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

    Type *NewVecTy = VectorType::get(Type::getInt64Ty(C), 2);

    Value *BC0 = Builder.CreateBitCast(Arg0, NewVecTy, "cast");
    Value *BC1 = Builder.CreateBitCast(Arg1, NewVecTy, "cast");

    CallInst *NewCall = Builder.CreateCall(NewFn, {BC0, BC1}, Name);
    CI->replaceAllUsesWith(NewCall);
    CI->eraseFromParent();
    return;
  }

  case Intrinsic::x86_sse41_insertps:
  case Intrinsic::x86_sse41_dppd:
  case Intrinsic::x86_sse41_dpps:
  case Intrinsic::x86_sse41_mpsadbw:
  case Intrinsic::x86_avx_dp_ps_256:
  case Intrinsic::x86_avx2_mpsadbw: {
    // Need to truncate the last argument from i32 to i8 -- this argument models
    // an inherently 8-bit immediate operand to these x86 instructions.
    SmallVector<Value *, 4> Args(CI->arg_operands().begin(),
                                 CI->arg_operands().end());

    // Replace the last argument with a trunc.
    Args.back() = Builder.CreateTrunc(Args.back(), Type::getInt8Ty(C), "trunc");

    CallInst *NewCall = Builder.CreateCall(NewFn, Args);
    CI->replaceAllUsesWith(NewCall);
    CI->eraseFromParent();
    return;
  }

  case Intrinsic::thread_pointer: {
    CI->replaceAllUsesWith(Builder.CreateCall(NewFn, {}));
    CI->eraseFromParent();
    return;
  }

  case Intrinsic::masked_load:
  case Intrinsic::masked_store: {
    SmallVector<Value *, 4> Args(CI->arg_operands().begin(),
                                 CI->arg_operands().end());
    CI->replaceAllUsesWith(Builder.CreateCall(NewFn, Args));
    CI->eraseFromParent();
    return;
  }
  }
}

void llvm::UpgradeCallsToIntrinsic(Function *F) {
  assert(F && "Illegal attempt to upgrade a non-existent intrinsic.");

  // Check if this function should be upgraded and get the replacement function
  // if there is one.
  Function *NewFn;
  if (UpgradeIntrinsicFunction(F, NewFn)) {
    // Replace all users of the old function with the new function or new
    // instructions. This is not a range loop because the call is deleted.
    for (auto UI = F->user_begin(), UE = F->user_end(); UI != UE; )
      if (CallInst *CI = dyn_cast<CallInst>(*UI++))
        UpgradeIntrinsicCall(CI, NewFn);

    // Remove old function, no longer used, from the module.
    F->eraseFromParent();
  }
}

void llvm::UpgradeInstWithTBAATag(Instruction *I) {
  MDNode *MD = I->getMetadata(LLVMContext::MD_tbaa);
  assert(MD && "UpgradeInstWithTBAATag should have a TBAA tag");
  // Check if the tag uses struct-path aware TBAA format.
  if (isa<MDNode>(MD->getOperand(0)) && MD->getNumOperands() >= 3)
    return;

  if (MD->getNumOperands() == 3) {
    Metadata *Elts[] = {MD->getOperand(0), MD->getOperand(1)};
    MDNode *ScalarType = MDNode::get(I->getContext(), Elts);
    // Create a MDNode <ScalarType, ScalarType, offset 0, const>
    Metadata *Elts2[] = {ScalarType, ScalarType,
                         ConstantAsMetadata::get(Constant::getNullValue(
                             Type::getInt64Ty(I->getContext()))),
                         MD->getOperand(2)};
    I->setMetadata(LLVMContext::MD_tbaa, MDNode::get(I->getContext(), Elts2));
  } else {
    // Create a MDNode <MD, MD, offset 0>
    Metadata *Elts[] = {MD, MD, ConstantAsMetadata::get(Constant::getNullValue(
                                    Type::getInt64Ty(I->getContext())))};
    I->setMetadata(LLVMContext::MD_tbaa, MDNode::get(I->getContext(), Elts));
  }
}

Instruction *llvm::UpgradeBitCastInst(unsigned Opc, Value *V, Type *DestTy,
                                      Instruction *&Temp) {
  if (Opc != Instruction::BitCast)
    return nullptr;

  Temp = nullptr;
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

  return nullptr;
}

Value *llvm::UpgradeBitCastExpr(unsigned Opc, Constant *C, Type *DestTy) {
  if (Opc != Instruction::BitCast)
    return nullptr;

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

  return nullptr;
}

/// Check the debug info version number, if it is out-dated, drop the debug
/// info. Return true if module is modified.
bool llvm::UpgradeDebugInfo(Module &M) {
  unsigned Version = getDebugMetadataVersionFromModule(M);
  if (Version == DEBUG_METADATA_VERSION)
    return false;

  bool RetCode = StripDebugInfo(M);
  if (RetCode) {
    DiagnosticInfoDebugMetadataVersion DiagVersion(M, Version);
    M.getContext().diagnose(DiagVersion);
  }
  return RetCode;
}

bool llvm::UpgradeModuleFlags(Module &M) {
  const NamedMDNode *ModFlags = M.getModuleFlagsMetadata();
  if (!ModFlags)
    return false;

  bool HasObjCFlag = false, HasClassProperties = false;
  for (unsigned I = 0, E = ModFlags->getNumOperands(); I != E; ++I) {
    MDNode *Op = ModFlags->getOperand(I);
    if (Op->getNumOperands() < 2)
      continue;
    MDString *ID = dyn_cast_or_null<MDString>(Op->getOperand(1));
    if (!ID)
      continue;
    if (ID->getString() == "Objective-C Image Info Version")
      HasObjCFlag = true;
    if (ID->getString() == "Objective-C Class Properties")
      HasClassProperties = true;
  }
  // "Objective-C Class Properties" is recently added for Objective-C. We
  // upgrade ObjC bitcodes to contain a "Objective-C Class Properties" module
  // flag of value 0, so we can correclty report error when trying to link
  // an ObjC bitcode without this module flag with an ObjC bitcode with this
  // module flag.
  if (HasObjCFlag && !HasClassProperties) {
    M.addModuleFlag(llvm::Module::Error, "Objective-C Class Properties",
                    (uint32_t)0);
    return true;
  }
  return false;
}

static bool isOldLoopArgument(Metadata *MD) {
  auto *T = dyn_cast_or_null<MDTuple>(MD);
  if (!T)
    return false;
  if (T->getNumOperands() < 1)
    return false;
  auto *S = dyn_cast_or_null<MDString>(T->getOperand(0));
  if (!S)
    return false;
  return S->getString().startswith("llvm.vectorizer.");
}

static MDString *upgradeLoopTag(LLVMContext &C, StringRef OldTag) {
  StringRef OldPrefix = "llvm.vectorizer.";
  assert(OldTag.startswith(OldPrefix) && "Expected old prefix");

  if (OldTag == "llvm.vectorizer.unroll")
    return MDString::get(C, "llvm.loop.interleave.count");

  return MDString::get(
      C, (Twine("llvm.loop.vectorize.") + OldTag.drop_front(OldPrefix.size()))
             .str());
}

static Metadata *upgradeLoopArgument(Metadata *MD) {
  auto *T = dyn_cast_or_null<MDTuple>(MD);
  if (!T)
    return MD;
  if (T->getNumOperands() < 1)
    return MD;
  auto *OldTag = dyn_cast_or_null<MDString>(T->getOperand(0));
  if (!OldTag)
    return MD;
  if (!OldTag->getString().startswith("llvm.vectorizer."))
    return MD;

  // This has an old tag.  Upgrade it.
  SmallVector<Metadata *, 8> Ops;
  Ops.reserve(T->getNumOperands());
  Ops.push_back(upgradeLoopTag(T->getContext(), OldTag->getString()));
  for (unsigned I = 1, E = T->getNumOperands(); I != E; ++I)
    Ops.push_back(T->getOperand(I));

  return MDTuple::get(T->getContext(), Ops);
}

MDNode *llvm::upgradeInstructionLoopAttachment(MDNode &N) {
  auto *T = dyn_cast<MDTuple>(&N);
  if (!T)
    return &N;

  if (!llvm::any_of(T->operands(), isOldLoopArgument))
    return &N;

  SmallVector<Metadata *, 8> Ops;
  Ops.reserve(T->getNumOperands());
  for (Metadata *MD : T->operands())
    Ops.push_back(upgradeLoopArgument(MD));

  return MDTuple::get(T->getContext(), Ops);
}
