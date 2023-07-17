//===-- AutoUpgrade.cpp - Implement auto-upgrade helper functions ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the auto-upgrade helper functions.
// This is where deprecated IR intrinsics and other IR features are updated to
// current specifications.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/AutoUpgrade.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/AttributeMask.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/IntrinsicsARM.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/IntrinsicsWebAssembly.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Regex.h"
#include "llvm/TargetParser/Triple.h"
#include <cstring>

using namespace llvm;

static cl::opt<bool>
    DisableAutoUpgradeDebugInfo("disable-auto-upgrade-debug-info",
                                cl::desc("Disable autoupgrade of debug info"));

static void rename(GlobalValue *GV) { GV->setName(GV->getName() + ".old"); }

// Upgrade the declarations of the SSE4.1 ptest intrinsics whose arguments have
// changed their type from v4f32 to v2i64.
static bool UpgradePTESTIntrinsic(Function* F, Intrinsic::ID IID,
                                  Function *&NewFn) {
  // Check whether this is an old version of the function, which received
  // v4f32 arguments.
  Type *Arg0Type = F->getFunctionType()->getParamType(0);
  if (Arg0Type != FixedVectorType::get(Type::getFloatTy(F->getContext()), 4))
    return false;

  // Yes, it's old, replace it with new version.
  rename(F);
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
  rename(F);
  NewFn = Intrinsic::getDeclaration(F->getParent(), IID);
  return true;
}

// Upgrade the declaration of fp compare intrinsics that change return type
// from scalar to vXi1 mask.
static bool UpgradeX86MaskedFPCompare(Function *F, Intrinsic::ID IID,
                                      Function *&NewFn) {
  // Check if the return type is a vector.
  if (F->getReturnType()->isVectorTy())
    return false;

  rename(F);
  NewFn = Intrinsic::getDeclaration(F->getParent(), IID);
  return true;
}

static bool UpgradeX86BF16Intrinsic(Function *F, Intrinsic::ID IID,
                                    Function *&NewFn) {
  if (F->getReturnType()->getScalarType()->isBFloatTy())
    return false;

  rename(F);
  NewFn = Intrinsic::getDeclaration(F->getParent(), IID);
  return true;
}

static bool UpgradeX86BF16DPIntrinsic(Function *F, Intrinsic::ID IID,
                                      Function *&NewFn) {
  if (F->getFunctionType()->getParamType(1)->getScalarType()->isBFloatTy())
    return false;

  rename(F);
  NewFn = Intrinsic::getDeclaration(F->getParent(), IID);
  return true;
}

static bool ShouldUpgradeX86Intrinsic(Function *F, StringRef Name) {
  // All of the intrinsics matches below should be marked with which llvm
  // version started autoupgrading them. At some point in the future we would
  // like to use this information to remove upgrade code for some older
  // intrinsics. It is currently undecided how we will determine that future
  // point.
  if (Name == "addcarryx.u32" || // Added in 8.0
      Name == "addcarryx.u64" || // Added in 8.0
      Name == "addcarry.u32" || // Added in 8.0
      Name == "addcarry.u64" || // Added in 8.0
      Name == "subborrow.u32" || // Added in 8.0
      Name == "subborrow.u64" || // Added in 8.0
      Name.startswith("sse2.padds.") || // Added in 8.0
      Name.startswith("sse2.psubs.") || // Added in 8.0
      Name.startswith("sse2.paddus.") || // Added in 8.0
      Name.startswith("sse2.psubus.") || // Added in 8.0
      Name.startswith("avx2.padds.") || // Added in 8.0
      Name.startswith("avx2.psubs.") || // Added in 8.0
      Name.startswith("avx2.paddus.") || // Added in 8.0
      Name.startswith("avx2.psubus.") || // Added in 8.0
      Name.startswith("avx512.padds.") || // Added in 8.0
      Name.startswith("avx512.psubs.") || // Added in 8.0
      Name.startswith("avx512.mask.padds.") || // Added in 8.0
      Name.startswith("avx512.mask.psubs.") || // Added in 8.0
      Name.startswith("avx512.mask.paddus.") || // Added in 8.0
      Name.startswith("avx512.mask.psubus.") || // Added in 8.0
      Name=="ssse3.pabs.b.128" || // Added in 6.0
      Name=="ssse3.pabs.w.128" || // Added in 6.0
      Name=="ssse3.pabs.d.128" || // Added in 6.0
      Name.startswith("fma4.vfmadd.s") || // Added in 7.0
      Name.startswith("fma.vfmadd.") || // Added in 7.0
      Name.startswith("fma.vfmsub.") || // Added in 7.0
      Name.startswith("fma.vfmsubadd.") || // Added in 7.0
      Name.startswith("fma.vfnmadd.") || // Added in 7.0
      Name.startswith("fma.vfnmsub.") || // Added in 7.0
      Name.startswith("avx512.mask.vfmadd.") || // Added in 7.0
      Name.startswith("avx512.mask.vfnmadd.") || // Added in 7.0
      Name.startswith("avx512.mask.vfnmsub.") || // Added in 7.0
      Name.startswith("avx512.mask3.vfmadd.") || // Added in 7.0
      Name.startswith("avx512.maskz.vfmadd.") || // Added in 7.0
      Name.startswith("avx512.mask3.vfmsub.") || // Added in 7.0
      Name.startswith("avx512.mask3.vfnmsub.") || // Added in 7.0
      Name.startswith("avx512.mask.vfmaddsub.") || // Added in 7.0
      Name.startswith("avx512.maskz.vfmaddsub.") || // Added in 7.0
      Name.startswith("avx512.mask3.vfmaddsub.") || // Added in 7.0
      Name.startswith("avx512.mask3.vfmsubadd.") || // Added in 7.0
      Name.startswith("avx512.mask.shuf.i") || // Added in 6.0
      Name.startswith("avx512.mask.shuf.f") || // Added in 6.0
      Name.startswith("avx512.kunpck") || //added in 6.0
      Name.startswith("avx2.pabs.") || // Added in 6.0
      Name.startswith("avx512.mask.pabs.") || // Added in 6.0
      Name.startswith("avx512.broadcastm") || // Added in 6.0
      Name == "sse.sqrt.ss" || // Added in 7.0
      Name == "sse2.sqrt.sd" || // Added in 7.0
      Name.startswith("avx512.mask.sqrt.p") || // Added in 7.0
      Name.startswith("avx.sqrt.p") || // Added in 7.0
      Name.startswith("sse2.sqrt.p") || // Added in 7.0
      Name.startswith("sse.sqrt.p") || // Added in 7.0
      Name.startswith("avx512.mask.pbroadcast") || // Added in 6.0
      Name.startswith("sse2.pcmpeq.") || // Added in 3.1
      Name.startswith("sse2.pcmpgt.") || // Added in 3.1
      Name.startswith("avx2.pcmpeq.") || // Added in 3.1
      Name.startswith("avx2.pcmpgt.") || // Added in 3.1
      Name.startswith("avx512.mask.pcmpeq.") || // Added in 3.9
      Name.startswith("avx512.mask.pcmpgt.") || // Added in 3.9
      Name.startswith("avx.vperm2f128.") || // Added in 6.0
      Name == "avx2.vperm2i128" || // Added in 6.0
      Name == "sse.add.ss" || // Added in 4.0
      Name == "sse2.add.sd" || // Added in 4.0
      Name == "sse.sub.ss" || // Added in 4.0
      Name == "sse2.sub.sd" || // Added in 4.0
      Name == "sse.mul.ss" || // Added in 4.0
      Name == "sse2.mul.sd" || // Added in 4.0
      Name == "sse.div.ss" || // Added in 4.0
      Name == "sse2.div.sd" || // Added in 4.0
      Name == "sse41.pmaxsb" || // Added in 3.9
      Name == "sse2.pmaxs.w" || // Added in 3.9
      Name == "sse41.pmaxsd" || // Added in 3.9
      Name == "sse2.pmaxu.b" || // Added in 3.9
      Name == "sse41.pmaxuw" || // Added in 3.9
      Name == "sse41.pmaxud" || // Added in 3.9
      Name == "sse41.pminsb" || // Added in 3.9
      Name == "sse2.pmins.w" || // Added in 3.9
      Name == "sse41.pminsd" || // Added in 3.9
      Name == "sse2.pminu.b" || // Added in 3.9
      Name == "sse41.pminuw" || // Added in 3.9
      Name == "sse41.pminud" || // Added in 3.9
      Name == "avx512.kand.w" || // Added in 7.0
      Name == "avx512.kandn.w" || // Added in 7.0
      Name == "avx512.knot.w" || // Added in 7.0
      Name == "avx512.kor.w" || // Added in 7.0
      Name == "avx512.kxor.w" || // Added in 7.0
      Name == "avx512.kxnor.w" || // Added in 7.0
      Name == "avx512.kortestc.w" || // Added in 7.0
      Name == "avx512.kortestz.w" || // Added in 7.0
      Name.startswith("avx512.mask.pshuf.b.") || // Added in 4.0
      Name.startswith("avx2.pmax") || // Added in 3.9
      Name.startswith("avx2.pmin") || // Added in 3.9
      Name.startswith("avx512.mask.pmax") || // Added in 4.0
      Name.startswith("avx512.mask.pmin") || // Added in 4.0
      Name.startswith("avx2.vbroadcast") || // Added in 3.8
      Name.startswith("avx2.pbroadcast") || // Added in 3.8
      Name.startswith("avx.vpermil.") || // Added in 3.1
      Name.startswith("sse2.pshuf") || // Added in 3.9
      Name.startswith("avx512.pbroadcast") || // Added in 3.9
      Name.startswith("avx512.mask.broadcast.s") || // Added in 3.9
      Name.startswith("avx512.mask.movddup") || // Added in 3.9
      Name.startswith("avx512.mask.movshdup") || // Added in 3.9
      Name.startswith("avx512.mask.movsldup") || // Added in 3.9
      Name.startswith("avx512.mask.pshuf.d.") || // Added in 3.9
      Name.startswith("avx512.mask.pshufl.w.") || // Added in 3.9
      Name.startswith("avx512.mask.pshufh.w.") || // Added in 3.9
      Name.startswith("avx512.mask.shuf.p") || // Added in 4.0
      Name.startswith("avx512.mask.vpermil.p") || // Added in 3.9
      Name.startswith("avx512.mask.perm.df.") || // Added in 3.9
      Name.startswith("avx512.mask.perm.di.") || // Added in 3.9
      Name.startswith("avx512.mask.punpckl") || // Added in 3.9
      Name.startswith("avx512.mask.punpckh") || // Added in 3.9
      Name.startswith("avx512.mask.unpckl.") || // Added in 3.9
      Name.startswith("avx512.mask.unpckh.") || // Added in 3.9
      Name.startswith("avx512.mask.pand.") || // Added in 3.9
      Name.startswith("avx512.mask.pandn.") || // Added in 3.9
      Name.startswith("avx512.mask.por.") || // Added in 3.9
      Name.startswith("avx512.mask.pxor.") || // Added in 3.9
      Name.startswith("avx512.mask.and.") || // Added in 3.9
      Name.startswith("avx512.mask.andn.") || // Added in 3.9
      Name.startswith("avx512.mask.or.") || // Added in 3.9
      Name.startswith("avx512.mask.xor.") || // Added in 3.9
      Name.startswith("avx512.mask.padd.") || // Added in 4.0
      Name.startswith("avx512.mask.psub.") || // Added in 4.0
      Name.startswith("avx512.mask.pmull.") || // Added in 4.0
      Name.startswith("avx512.mask.cvtdq2pd.") || // Added in 4.0
      Name.startswith("avx512.mask.cvtudq2pd.") || // Added in 4.0
      Name.startswith("avx512.mask.cvtudq2ps.") || // Added in 7.0 updated 9.0
      Name.startswith("avx512.mask.cvtqq2pd.") || // Added in 7.0 updated 9.0
      Name.startswith("avx512.mask.cvtuqq2pd.") || // Added in 7.0 updated 9.0
      Name.startswith("avx512.mask.cvtdq2ps.") || // Added in 7.0 updated 9.0
      Name == "avx512.mask.vcvtph2ps.128" || // Added in 11.0
      Name == "avx512.mask.vcvtph2ps.256" || // Added in 11.0
      Name == "avx512.mask.cvtqq2ps.256" || // Added in 9.0
      Name == "avx512.mask.cvtqq2ps.512" || // Added in 9.0
      Name == "avx512.mask.cvtuqq2ps.256" || // Added in 9.0
      Name == "avx512.mask.cvtuqq2ps.512" || // Added in 9.0
      Name == "avx512.mask.cvtpd2dq.256" || // Added in 7.0
      Name == "avx512.mask.cvtpd2ps.256" || // Added in 7.0
      Name == "avx512.mask.cvttpd2dq.256" || // Added in 7.0
      Name == "avx512.mask.cvttps2dq.128" || // Added in 7.0
      Name == "avx512.mask.cvttps2dq.256" || // Added in 7.0
      Name == "avx512.mask.cvtps2pd.128" || // Added in 7.0
      Name == "avx512.mask.cvtps2pd.256" || // Added in 7.0
      Name == "avx512.cvtusi2sd" || // Added in 7.0
      Name.startswith("avx512.mask.permvar.") || // Added in 7.0
      Name == "sse2.pmulu.dq" || // Added in 7.0
      Name == "sse41.pmuldq" || // Added in 7.0
      Name == "avx2.pmulu.dq" || // Added in 7.0
      Name == "avx2.pmul.dq" || // Added in 7.0
      Name == "avx512.pmulu.dq.512" || // Added in 7.0
      Name == "avx512.pmul.dq.512" || // Added in 7.0
      Name.startswith("avx512.mask.pmul.dq.") || // Added in 4.0
      Name.startswith("avx512.mask.pmulu.dq.") || // Added in 4.0
      Name.startswith("avx512.mask.pmul.hr.sw.") || // Added in 7.0
      Name.startswith("avx512.mask.pmulh.w.") || // Added in 7.0
      Name.startswith("avx512.mask.pmulhu.w.") || // Added in 7.0
      Name.startswith("avx512.mask.pmaddw.d.") || // Added in 7.0
      Name.startswith("avx512.mask.pmaddubs.w.") || // Added in 7.0
      Name.startswith("avx512.mask.packsswb.") || // Added in 5.0
      Name.startswith("avx512.mask.packssdw.") || // Added in 5.0
      Name.startswith("avx512.mask.packuswb.") || // Added in 5.0
      Name.startswith("avx512.mask.packusdw.") || // Added in 5.0
      Name.startswith("avx512.mask.cmp.b") || // Added in 5.0
      Name.startswith("avx512.mask.cmp.d") || // Added in 5.0
      Name.startswith("avx512.mask.cmp.q") || // Added in 5.0
      Name.startswith("avx512.mask.cmp.w") || // Added in 5.0
      Name.startswith("avx512.cmp.p") || // Added in 12.0
      Name.startswith("avx512.mask.ucmp.") || // Added in 5.0
      Name.startswith("avx512.cvtb2mask.") || // Added in 7.0
      Name.startswith("avx512.cvtw2mask.") || // Added in 7.0
      Name.startswith("avx512.cvtd2mask.") || // Added in 7.0
      Name.startswith("avx512.cvtq2mask.") || // Added in 7.0
      Name.startswith("avx512.mask.vpermilvar.") || // Added in 4.0
      Name.startswith("avx512.mask.psll.d") || // Added in 4.0
      Name.startswith("avx512.mask.psll.q") || // Added in 4.0
      Name.startswith("avx512.mask.psll.w") || // Added in 4.0
      Name.startswith("avx512.mask.psra.d") || // Added in 4.0
      Name.startswith("avx512.mask.psra.q") || // Added in 4.0
      Name.startswith("avx512.mask.psra.w") || // Added in 4.0
      Name.startswith("avx512.mask.psrl.d") || // Added in 4.0
      Name.startswith("avx512.mask.psrl.q") || // Added in 4.0
      Name.startswith("avx512.mask.psrl.w") || // Added in 4.0
      Name.startswith("avx512.mask.pslli") || // Added in 4.0
      Name.startswith("avx512.mask.psrai") || // Added in 4.0
      Name.startswith("avx512.mask.psrli") || // Added in 4.0
      Name.startswith("avx512.mask.psllv") || // Added in 4.0
      Name.startswith("avx512.mask.psrav") || // Added in 4.0
      Name.startswith("avx512.mask.psrlv") || // Added in 4.0
      Name.startswith("sse41.pmovsx") || // Added in 3.8
      Name.startswith("sse41.pmovzx") || // Added in 3.9
      Name.startswith("avx2.pmovsx") || // Added in 3.9
      Name.startswith("avx2.pmovzx") || // Added in 3.9
      Name.startswith("avx512.mask.pmovsx") || // Added in 4.0
      Name.startswith("avx512.mask.pmovzx") || // Added in 4.0
      Name.startswith("avx512.mask.lzcnt.") || // Added in 5.0
      Name.startswith("avx512.mask.pternlog.") || // Added in 7.0
      Name.startswith("avx512.maskz.pternlog.") || // Added in 7.0
      Name.startswith("avx512.mask.vpmadd52") || // Added in 7.0
      Name.startswith("avx512.maskz.vpmadd52") || // Added in 7.0
      Name.startswith("avx512.mask.vpermi2var.") || // Added in 7.0
      Name.startswith("avx512.mask.vpermt2var.") || // Added in 7.0
      Name.startswith("avx512.maskz.vpermt2var.") || // Added in 7.0
      Name.startswith("avx512.mask.vpdpbusd.") || // Added in 7.0
      Name.startswith("avx512.maskz.vpdpbusd.") || // Added in 7.0
      Name.startswith("avx512.mask.vpdpbusds.") || // Added in 7.0
      Name.startswith("avx512.maskz.vpdpbusds.") || // Added in 7.0
      Name.startswith("avx512.mask.vpdpwssd.") || // Added in 7.0
      Name.startswith("avx512.maskz.vpdpwssd.") || // Added in 7.0
      Name.startswith("avx512.mask.vpdpwssds.") || // Added in 7.0
      Name.startswith("avx512.maskz.vpdpwssds.") || // Added in 7.0
      Name.startswith("avx512.mask.dbpsadbw.") || // Added in 7.0
      Name.startswith("avx512.mask.vpshld.") || // Added in 7.0
      Name.startswith("avx512.mask.vpshrd.") || // Added in 7.0
      Name.startswith("avx512.mask.vpshldv.") || // Added in 8.0
      Name.startswith("avx512.mask.vpshrdv.") || // Added in 8.0
      Name.startswith("avx512.maskz.vpshldv.") || // Added in 8.0
      Name.startswith("avx512.maskz.vpshrdv.") || // Added in 8.0
      Name.startswith("avx512.vpshld.") || // Added in 8.0
      Name.startswith("avx512.vpshrd.") || // Added in 8.0
      Name.startswith("avx512.mask.add.p") || // Added in 7.0. 128/256 in 4.0
      Name.startswith("avx512.mask.sub.p") || // Added in 7.0. 128/256 in 4.0
      Name.startswith("avx512.mask.mul.p") || // Added in 7.0. 128/256 in 4.0
      Name.startswith("avx512.mask.div.p") || // Added in 7.0. 128/256 in 4.0
      Name.startswith("avx512.mask.max.p") || // Added in 7.0. 128/256 in 5.0
      Name.startswith("avx512.mask.min.p") || // Added in 7.0. 128/256 in 5.0
      Name.startswith("avx512.mask.fpclass.p") || // Added in 7.0
      Name.startswith("avx512.mask.vpshufbitqmb.") || // Added in 8.0
      Name.startswith("avx512.mask.pmultishift.qb.") || // Added in 8.0
      Name.startswith("avx512.mask.conflict.") || // Added in 9.0
      Name == "avx512.mask.pmov.qd.256" || // Added in 9.0
      Name == "avx512.mask.pmov.qd.512" || // Added in 9.0
      Name == "avx512.mask.pmov.wb.256" || // Added in 9.0
      Name == "avx512.mask.pmov.wb.512" || // Added in 9.0
      Name == "sse.cvtsi2ss" || // Added in 7.0
      Name == "sse.cvtsi642ss" || // Added in 7.0
      Name == "sse2.cvtsi2sd" || // Added in 7.0
      Name == "sse2.cvtsi642sd" || // Added in 7.0
      Name == "sse2.cvtss2sd" || // Added in 7.0
      Name == "sse2.cvtdq2pd" || // Added in 3.9
      Name == "sse2.cvtdq2ps" || // Added in 7.0
      Name == "sse2.cvtps2pd" || // Added in 3.9
      Name == "avx.cvtdq2.pd.256" || // Added in 3.9
      Name == "avx.cvtdq2.ps.256" || // Added in 7.0
      Name == "avx.cvt.ps2.pd.256" || // Added in 3.9
      Name.startswith("vcvtph2ps.") || // Added in 11.0
      Name.startswith("avx.vinsertf128.") || // Added in 3.7
      Name == "avx2.vinserti128" || // Added in 3.7
      Name.startswith("avx512.mask.insert") || // Added in 4.0
      Name.startswith("avx.vextractf128.") || // Added in 3.7
      Name == "avx2.vextracti128" || // Added in 3.7
      Name.startswith("avx512.mask.vextract") || // Added in 4.0
      Name.startswith("sse4a.movnt.") || // Added in 3.9
      Name.startswith("avx.movnt.") || // Added in 3.2
      Name.startswith("avx512.storent.") || // Added in 3.9
      Name == "sse41.movntdqa" || // Added in 5.0
      Name == "avx2.movntdqa" || // Added in 5.0
      Name == "avx512.movntdqa" || // Added in 5.0
      Name == "sse2.storel.dq" || // Added in 3.9
      Name.startswith("sse.storeu.") || // Added in 3.9
      Name.startswith("sse2.storeu.") || // Added in 3.9
      Name.startswith("avx.storeu.") || // Added in 3.9
      Name.startswith("avx512.mask.storeu.") || // Added in 3.9
      Name.startswith("avx512.mask.store.p") || // Added in 3.9
      Name.startswith("avx512.mask.store.b.") || // Added in 3.9
      Name.startswith("avx512.mask.store.w.") || // Added in 3.9
      Name.startswith("avx512.mask.store.d.") || // Added in 3.9
      Name.startswith("avx512.mask.store.q.") || // Added in 3.9
      Name == "avx512.mask.store.ss" || // Added in 7.0
      Name.startswith("avx512.mask.loadu.") || // Added in 3.9
      Name.startswith("avx512.mask.load.") || // Added in 3.9
      Name.startswith("avx512.mask.expand.load.") || // Added in 7.0
      Name.startswith("avx512.mask.compress.store.") || // Added in 7.0
      Name.startswith("avx512.mask.expand.b") || // Added in 9.0
      Name.startswith("avx512.mask.expand.w") || // Added in 9.0
      Name.startswith("avx512.mask.expand.d") || // Added in 9.0
      Name.startswith("avx512.mask.expand.q") || // Added in 9.0
      Name.startswith("avx512.mask.expand.p") || // Added in 9.0
      Name.startswith("avx512.mask.compress.b") || // Added in 9.0
      Name.startswith("avx512.mask.compress.w") || // Added in 9.0
      Name.startswith("avx512.mask.compress.d") || // Added in 9.0
      Name.startswith("avx512.mask.compress.q") || // Added in 9.0
      Name.startswith("avx512.mask.compress.p") || // Added in 9.0
      Name == "sse42.crc32.64.8" || // Added in 3.4
      Name.startswith("avx.vbroadcast.s") || // Added in 3.5
      Name.startswith("avx512.vbroadcast.s") || // Added in 7.0
      Name.startswith("avx512.mask.palignr.") || // Added in 3.9
      Name.startswith("avx512.mask.valign.") || // Added in 4.0
      Name.startswith("sse2.psll.dq") || // Added in 3.7
      Name.startswith("sse2.psrl.dq") || // Added in 3.7
      Name.startswith("avx2.psll.dq") || // Added in 3.7
      Name.startswith("avx2.psrl.dq") || // Added in 3.7
      Name.startswith("avx512.psll.dq") || // Added in 3.9
      Name.startswith("avx512.psrl.dq") || // Added in 3.9
      Name == "sse41.pblendw" || // Added in 3.7
      Name.startswith("sse41.blendp") || // Added in 3.7
      Name.startswith("avx.blend.p") || // Added in 3.7
      Name == "avx2.pblendw" || // Added in 3.7
      Name.startswith("avx2.pblendd.") || // Added in 3.7
      Name.startswith("avx.vbroadcastf128") || // Added in 4.0
      Name == "avx2.vbroadcasti128" || // Added in 3.7
      Name.startswith("avx512.mask.broadcastf32x4.") || // Added in 6.0
      Name.startswith("avx512.mask.broadcastf64x2.") || // Added in 6.0
      Name.startswith("avx512.mask.broadcastf32x8.") || // Added in 6.0
      Name.startswith("avx512.mask.broadcastf64x4.") || // Added in 6.0
      Name.startswith("avx512.mask.broadcasti32x4.") || // Added in 6.0
      Name.startswith("avx512.mask.broadcasti64x2.") || // Added in 6.0
      Name.startswith("avx512.mask.broadcasti32x8.") || // Added in 6.0
      Name.startswith("avx512.mask.broadcasti64x4.") || // Added in 6.0
      Name == "xop.vpcmov" || // Added in 3.8
      Name == "xop.vpcmov.256" || // Added in 5.0
      Name.startswith("avx512.mask.move.s") || // Added in 4.0
      Name.startswith("avx512.cvtmask2") || // Added in 5.0
      Name.startswith("xop.vpcom") || // Added in 3.2, Updated in 9.0
      Name.startswith("xop.vprot") || // Added in 8.0
      Name.startswith("avx512.prol") || // Added in 8.0
      Name.startswith("avx512.pror") || // Added in 8.0
      Name.startswith("avx512.mask.prorv.") || // Added in 8.0
      Name.startswith("avx512.mask.pror.") ||  // Added in 8.0
      Name.startswith("avx512.mask.prolv.") || // Added in 8.0
      Name.startswith("avx512.mask.prol.") ||  // Added in 8.0
      Name.startswith("avx512.ptestm") || //Added in 6.0
      Name.startswith("avx512.ptestnm") || //Added in 6.0
      Name.startswith("avx512.mask.pavg")) // Added in 6.0
    return true;

  return false;
}

static bool UpgradeX86IntrinsicFunction(Function *F, StringRef Name,
                                        Function *&NewFn) {
  // Only handle intrinsics that start with "x86.".
  if (!Name.startswith("x86."))
    return false;
  // Remove "x86." prefix.
  Name = Name.substr(4);

  if (ShouldUpgradeX86Intrinsic(F, Name)) {
    NewFn = nullptr;
    return true;
  }

  if (Name == "rdtscp") { // Added in 8.0
    // If this intrinsic has 0 operands, it's the new version.
    if (F->getFunctionType()->getNumParams() == 0)
      return false;

    rename(F);
    NewFn = Intrinsic::getDeclaration(F->getParent(),
                                      Intrinsic::x86_rdtscp);
    return true;
  }

  // SSE4.1 ptest functions may have an old signature.
  if (Name.startswith("sse41.ptest")) { // Added in 3.2
    if (Name.substr(11) == "c")
      return UpgradePTESTIntrinsic(F, Intrinsic::x86_sse41_ptestc, NewFn);
    if (Name.substr(11) == "z")
      return UpgradePTESTIntrinsic(F, Intrinsic::x86_sse41_ptestz, NewFn);
    if (Name.substr(11) == "nzc")
      return UpgradePTESTIntrinsic(F, Intrinsic::x86_sse41_ptestnzc, NewFn);
  }
  // Several blend and other instructions with masks used the wrong number of
  // bits.
  if (Name == "sse41.insertps") // Added in 3.6
    return UpgradeX86IntrinsicsWith8BitMask(F, Intrinsic::x86_sse41_insertps,
                                            NewFn);
  if (Name == "sse41.dppd") // Added in 3.6
    return UpgradeX86IntrinsicsWith8BitMask(F, Intrinsic::x86_sse41_dppd,
                                            NewFn);
  if (Name == "sse41.dpps") // Added in 3.6
    return UpgradeX86IntrinsicsWith8BitMask(F, Intrinsic::x86_sse41_dpps,
                                            NewFn);
  if (Name == "sse41.mpsadbw") // Added in 3.6
    return UpgradeX86IntrinsicsWith8BitMask(F, Intrinsic::x86_sse41_mpsadbw,
                                            NewFn);
  if (Name == "avx.dp.ps.256") // Added in 3.6
    return UpgradeX86IntrinsicsWith8BitMask(F, Intrinsic::x86_avx_dp_ps_256,
                                            NewFn);
  if (Name == "avx2.mpsadbw") // Added in 3.6
    return UpgradeX86IntrinsicsWith8BitMask(F, Intrinsic::x86_avx2_mpsadbw,
                                            NewFn);
  if (Name == "avx512.mask.cmp.pd.128") // Added in 7.0
    return UpgradeX86MaskedFPCompare(F, Intrinsic::x86_avx512_mask_cmp_pd_128,
                                     NewFn);
  if (Name == "avx512.mask.cmp.pd.256") // Added in 7.0
    return UpgradeX86MaskedFPCompare(F, Intrinsic::x86_avx512_mask_cmp_pd_256,
                                     NewFn);
  if (Name == "avx512.mask.cmp.pd.512") // Added in 7.0
    return UpgradeX86MaskedFPCompare(F, Intrinsic::x86_avx512_mask_cmp_pd_512,
                                     NewFn);
  if (Name == "avx512.mask.cmp.ps.128") // Added in 7.0
    return UpgradeX86MaskedFPCompare(F, Intrinsic::x86_avx512_mask_cmp_ps_128,
                                     NewFn);
  if (Name == "avx512.mask.cmp.ps.256") // Added in 7.0
    return UpgradeX86MaskedFPCompare(F, Intrinsic::x86_avx512_mask_cmp_ps_256,
                                     NewFn);
  if (Name == "avx512.mask.cmp.ps.512") // Added in 7.0
    return UpgradeX86MaskedFPCompare(F, Intrinsic::x86_avx512_mask_cmp_ps_512,
                                     NewFn);
  if (Name == "avx512bf16.cvtne2ps2bf16.128") // Added in 9.0
    return UpgradeX86BF16Intrinsic(
        F, Intrinsic::x86_avx512bf16_cvtne2ps2bf16_128, NewFn);
  if (Name == "avx512bf16.cvtne2ps2bf16.256") // Added in 9.0
    return UpgradeX86BF16Intrinsic(
        F, Intrinsic::x86_avx512bf16_cvtne2ps2bf16_256, NewFn);
  if (Name == "avx512bf16.cvtne2ps2bf16.512") // Added in 9.0
    return UpgradeX86BF16Intrinsic(
        F, Intrinsic::x86_avx512bf16_cvtne2ps2bf16_512, NewFn);
  if (Name == "avx512bf16.mask.cvtneps2bf16.128") // Added in 9.0
    return UpgradeX86BF16Intrinsic(
        F, Intrinsic::x86_avx512bf16_mask_cvtneps2bf16_128, NewFn);
  if (Name == "avx512bf16.cvtneps2bf16.256") // Added in 9.0
    return UpgradeX86BF16Intrinsic(
        F, Intrinsic::x86_avx512bf16_cvtneps2bf16_256, NewFn);
  if (Name == "avx512bf16.cvtneps2bf16.512") // Added in 9.0
    return UpgradeX86BF16Intrinsic(
        F, Intrinsic::x86_avx512bf16_cvtneps2bf16_512, NewFn);
  if (Name == "avx512bf16.dpbf16ps.128") // Added in 9.0
    return UpgradeX86BF16DPIntrinsic(
        F, Intrinsic::x86_avx512bf16_dpbf16ps_128, NewFn);
  if (Name == "avx512bf16.dpbf16ps.256") // Added in 9.0
    return UpgradeX86BF16DPIntrinsic(
        F, Intrinsic::x86_avx512bf16_dpbf16ps_256, NewFn);
  if (Name == "avx512bf16.dpbf16ps.512") // Added in 9.0
    return UpgradeX86BF16DPIntrinsic(
        F, Intrinsic::x86_avx512bf16_dpbf16ps_512, NewFn);

  // frcz.ss/sd may need to have an argument dropped. Added in 3.2
  if (Name.startswith("xop.vfrcz.ss") && F->arg_size() == 2) {
    rename(F);
    NewFn = Intrinsic::getDeclaration(F->getParent(),
                                      Intrinsic::x86_xop_vfrcz_ss);
    return true;
  }
  if (Name.startswith("xop.vfrcz.sd") && F->arg_size() == 2) {
    rename(F);
    NewFn = Intrinsic::getDeclaration(F->getParent(),
                                      Intrinsic::x86_xop_vfrcz_sd);
    return true;
  }
  // Upgrade any XOP PERMIL2 index operand still using a float/double vector.
  if (Name.startswith("xop.vpermil2")) { // Added in 3.9
    auto Idx = F->getFunctionType()->getParamType(2);
    if (Idx->isFPOrFPVectorTy()) {
      rename(F);
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

  if (Name == "seh.recoverfp") {
    NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::eh_recoverfp);
    return true;
  }

  return false;
}

static Intrinsic::ID ShouldUpgradeNVPTXBF16Intrinsic(StringRef Name) {
  return StringSwitch<Intrinsic::ID>(Name)
      .Case("abs.bf16", Intrinsic::nvvm_abs_bf16)
      .Case("abs.bf16x2", Intrinsic::nvvm_abs_bf16x2)
      .Case("fma.rn.bf16", Intrinsic::nvvm_fma_rn_bf16)
      .Case("fma.rn.bf16x2", Intrinsic::nvvm_fma_rn_bf16x2)
      .Case("fma.rn.ftz_bf16", Intrinsic::nvvm_fma_rn_ftz_bf16)
      .Case("fma.rn.ftz.bf16x2", Intrinsic::nvvm_fma_rn_ftz_bf16x2)
      .Case("fma.rn.ftz.relu.bf16", Intrinsic::nvvm_fma_rn_ftz_relu_bf16)
      .Case("fma.rn.ftz.relu.bf16x2", Intrinsic::nvvm_fma_rn_ftz_relu_bf16x2)
      .Case("fma.rn.ftz_sat.bf16", Intrinsic::nvvm_fma_rn_ftz_sat_bf16)
      .Case("fma.rn.ftz_sat.bf16x2", Intrinsic::nvvm_fma_rn_ftz_sat_bf16x2)
      .Case("fma.rn.relu.bf16", Intrinsic::nvvm_fma_rn_relu_bf16)
      .Case("fma.rn.relu.bf16x2", Intrinsic::nvvm_fma_rn_relu_bf16x2)
      .Case("fma.rn.sat.bf16", Intrinsic::nvvm_fma_rn_sat_bf16)
      .Case("fma.rn.sat.bf16x2", Intrinsic::nvvm_fma_rn_sat_bf16x2)
      .Case("fmax.bf16", Intrinsic::nvvm_fmax_bf16)
      .Case("fmax.bf16x2", Intrinsic::nvvm_fmax_bf16x2)
      .Case("fmax.ftz.bf16", Intrinsic::nvvm_fmax_ftz_bf16)
      .Case("fmax.ftz.bf16x2", Intrinsic::nvvm_fmax_ftz_bf16x2)
      .Case("fmax.ftz.nan.bf16", Intrinsic::nvvm_fmax_ftz_nan_bf16)
      .Case("fmax.ftz.nan.bf16x2", Intrinsic::nvvm_fmax_ftz_nan_bf16x2)
      .Case("fmax.ftz.nan.xorsign.abs.bf16",
            Intrinsic::nvvm_fmax_ftz_nan_xorsign_abs_bf16)
      .Case("fmax.ftz.nan.xorsign.abs.bf16x2",
            Intrinsic::nvvm_fmax_ftz_nan_xorsign_abs_bf16x2)
      .Case("fmax.ftz.xorsign.abs.bf16",
            Intrinsic::nvvm_fmax_ftz_xorsign_abs_bf16)
      .Case("fmax.ftz.xorsign.abs.bf16x2",
            Intrinsic::nvvm_fmax_ftz_xorsign_abs_bf16x2)
      .Case("fmax.nan.bf16", Intrinsic::nvvm_fmax_nan_bf16)
      .Case("fmax.nan.bf16x2", Intrinsic::nvvm_fmax_nan_bf16x2)
      .Case("fmax.nan.xorsign.abs.bf16",
            Intrinsic::nvvm_fmax_nan_xorsign_abs_bf16)
      .Case("fmax.nan.xorsign.abs.bf16x2",
            Intrinsic::nvvm_fmax_nan_xorsign_abs_bf16x2)
      .Case("fmax.xorsign.abs.bf16", Intrinsic::nvvm_fmax_xorsign_abs_bf16)
      .Case("fmax.xorsign.abs.bf16x2", Intrinsic::nvvm_fmax_xorsign_abs_bf16x2)
      .Case("fmin.bf16", Intrinsic::nvvm_fmin_bf16)
      .Case("fmin.bf16x2", Intrinsic::nvvm_fmin_bf16x2)
      .Case("fmin.ftz.bf16", Intrinsic::nvvm_fmin_ftz_bf16)
      .Case("fmin.ftz.bf16x2", Intrinsic::nvvm_fmin_ftz_bf16x2)
      .Case("fmin.ftz.nan_bf16", Intrinsic::nvvm_fmin_ftz_nan_bf16)
      .Case("fmin.ftz.nan_bf16x2", Intrinsic::nvvm_fmin_ftz_nan_bf16x2)
      .Case("fmin.ftz.nan.xorsign.abs.bf16",
            Intrinsic::nvvm_fmin_ftz_nan_xorsign_abs_bf16)
      .Case("fmin.ftz.nan.xorsign.abs.bf16x2",
            Intrinsic::nvvm_fmin_ftz_nan_xorsign_abs_bf16x2)
      .Case("fmin.ftz.xorsign.abs.bf16",
            Intrinsic::nvvm_fmin_ftz_xorsign_abs_bf16)
      .Case("fmin.ftz.xorsign.abs.bf16x2",
            Intrinsic::nvvm_fmin_ftz_xorsign_abs_bf16x2)
      .Case("fmin.nan.bf16", Intrinsic::nvvm_fmin_nan_bf16)
      .Case("fmin.nan.bf16x2", Intrinsic::nvvm_fmin_nan_bf16x2)
      .Case("fmin.nan.xorsign.abs.bf16",
            Intrinsic::nvvm_fmin_nan_xorsign_abs_bf16)
      .Case("fmin.nan.xorsign.abs.bf16x2",
            Intrinsic::nvvm_fmin_nan_xorsign_abs_bf16x2)
      .Case("fmin.xorsign.abs.bf16", Intrinsic::nvvm_fmin_xorsign_abs_bf16)
      .Case("fmin.xorsign.abs.bf16x2", Intrinsic::nvvm_fmin_xorsign_abs_bf16x2)
      .Case("neg.bf16", Intrinsic::nvvm_neg_bf16)
      .Case("neg.bf16x2", Intrinsic::nvvm_neg_bf16x2)
      .Default(Intrinsic::not_intrinsic);
}

static bool UpgradeIntrinsicFunction1(Function *F, Function *&NewFn) {
  assert(F && "Illegal to upgrade a non-existent Function.");

  // Quickly eliminate it, if it's not a candidate.
  StringRef Name = F->getName();
  if (Name.size() <= 7 || !Name.startswith("llvm."))
    return false;
  Name = Name.substr(5); // Strip off "llvm."

  switch (Name[0]) {
  default: break;
  case 'a': {
    if (Name.startswith("arm.rbit") || Name.startswith("aarch64.rbit")) {
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::bitreverse,
                                        F->arg_begin()->getType());
      return true;
    }
    if (Name.startswith("aarch64.neon.frintn")) {
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::roundeven,
                                        F->arg_begin()->getType());
      return true;
    }
    if (Name.startswith("aarch64.neon.rbit")) {
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::bitreverse,
                                        F->arg_begin()->getType());
      return true;
    }
    if (Name == "aarch64.sve.bfdot.lane") {
      NewFn = Intrinsic::getDeclaration(F->getParent(),
                                        Intrinsic::aarch64_sve_bfdot_lane_v2);
      return true;
    }
    if (Name == "aarch64.sve.bfmlalb.lane") {
      NewFn = Intrinsic::getDeclaration(F->getParent(),
                                        Intrinsic::aarch64_sve_bfmlalb_lane_v2);
      return true;
    }
    if (Name == "aarch64.sve.bfmlalt.lane") {
      NewFn = Intrinsic::getDeclaration(F->getParent(),
                                        Intrinsic::aarch64_sve_bfmlalt_lane_v2);
      return true;
    }
    static const Regex LdRegex("^aarch64\\.sve\\.ld[234](.nxv[a-z0-9]+|$)");
    if (LdRegex.match(Name)) {
      Type *ScalarTy =
          dyn_cast<VectorType>(F->getReturnType())->getElementType();
      ElementCount EC =
          dyn_cast<VectorType>(F->arg_begin()->getType())->getElementCount();
      Type *Ty = VectorType::get(ScalarTy, EC);
      Intrinsic::ID ID =
          StringSwitch<Intrinsic::ID>(Name)
              .StartsWith("aarch64.sve.ld2", Intrinsic::aarch64_sve_ld2_sret)
              .StartsWith("aarch64.sve.ld3", Intrinsic::aarch64_sve_ld3_sret)
              .StartsWith("aarch64.sve.ld4", Intrinsic::aarch64_sve_ld4_sret)
              .Default(Intrinsic::not_intrinsic);
      NewFn = Intrinsic::getDeclaration(F->getParent(), ID, Ty);
      return true;
    }
    if (Name.startswith("aarch64.sve.tuple.get")) {
      Type *Tys[] = {F->getReturnType(), F->arg_begin()->getType()};
      NewFn = Intrinsic::getDeclaration(F->getParent(),
                                        Intrinsic::vector_extract, Tys);
      return true;
    }
    if (Name.startswith("aarch64.sve.tuple.set")) {
      auto Args = F->getFunctionType()->params();
      Type *Tys[] = {Args[0], Args[2], Args[1]};
      NewFn = Intrinsic::getDeclaration(F->getParent(),
                                        Intrinsic::vector_insert, Tys);
      return true;
    }
    static const Regex CreateTupleRegex(
        "^aarch64\\.sve\\.tuple\\.create[234](.nxv[a-z0-9]+|$)");
    if (CreateTupleRegex.match(Name)) {
      auto Args = F->getFunctionType()->params();
      Type *Tys[] = {F->getReturnType(), Args[1]};
      NewFn = Intrinsic::getDeclaration(F->getParent(),
                                        Intrinsic::vector_insert, Tys);
      return true;
    }
    if (Name.startswith("arm.neon.vclz")) {
      Type* args[2] = {
        F->arg_begin()->getType(),
        Type::getInt1Ty(F->getContext())
      };
      // Can't use Intrinsic::getDeclaration here as it adds a ".i1" to
      // the end of the name. Change name from llvm.arm.neon.vclz.* to
      //  llvm.ctlz.*
      FunctionType* fType = FunctionType::get(F->getReturnType(), args, false);
      NewFn = Function::Create(fType, F->getLinkage(), F->getAddressSpace(),
                               "llvm.ctlz." + Name.substr(14), F->getParent());
      return true;
    }
    if (Name.startswith("arm.neon.vcnt")) {
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::ctpop,
                                        F->arg_begin()->getType());
      return true;
    }
    static const Regex vstRegex("^arm\\.neon\\.vst([1234]|[234]lane)\\.v[a-z0-9]*$");
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
      if (!Name.contains("lane"))
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
    if (Name.startswith("arm.neon.vqadds.")) {
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::sadd_sat,
                                        F->arg_begin()->getType());
      return true;
    }
    if (Name.startswith("arm.neon.vqaddu.")) {
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::uadd_sat,
                                        F->arg_begin()->getType());
      return true;
    }
    if (Name.startswith("arm.neon.vqsubs.")) {
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::ssub_sat,
                                        F->arg_begin()->getType());
      return true;
    }
    if (Name.startswith("arm.neon.vqsubu.")) {
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::usub_sat,
                                        F->arg_begin()->getType());
      return true;
    }
    if (Name.startswith("aarch64.neon.addp")) {
      if (F->arg_size() != 2)
        break; // Invalid IR.
      VectorType *Ty = dyn_cast<VectorType>(F->getReturnType());
      if (Ty && Ty->getElementType()->isFloatingPointTy()) {
        NewFn = Intrinsic::getDeclaration(F->getParent(),
                                          Intrinsic::aarch64_neon_faddp, Ty);
        return true;
      }
    }

    // Changed in 12.0: bfdot accept v4bf16 and v8bf16 instead of v8i8 and v16i8
    // respectively
    if ((Name.startswith("arm.neon.bfdot.") ||
         Name.startswith("aarch64.neon.bfdot.")) &&
        Name.endswith("i8")) {
      Intrinsic::ID IID =
          StringSwitch<Intrinsic::ID>(Name)
              .Cases("arm.neon.bfdot.v2f32.v8i8",
                     "arm.neon.bfdot.v4f32.v16i8",
                     Intrinsic::arm_neon_bfdot)
              .Cases("aarch64.neon.bfdot.v2f32.v8i8",
                     "aarch64.neon.bfdot.v4f32.v16i8",
                     Intrinsic::aarch64_neon_bfdot)
              .Default(Intrinsic::not_intrinsic);
      if (IID == Intrinsic::not_intrinsic)
        break;

      size_t OperandWidth = F->getReturnType()->getPrimitiveSizeInBits();
      assert((OperandWidth == 64 || OperandWidth == 128) &&
             "Unexpected operand width");
      LLVMContext &Ctx = F->getParent()->getContext();
      std::array<Type *, 2> Tys {{
        F->getReturnType(),
        FixedVectorType::get(Type::getBFloatTy(Ctx), OperandWidth / 16)
      }};
      NewFn = Intrinsic::getDeclaration(F->getParent(), IID, Tys);
      return true;
    }

    // Changed in 12.0: bfmmla, bfmlalb and bfmlalt are not polymorphic anymore
    // and accept v8bf16 instead of v16i8
    if ((Name.startswith("arm.neon.bfm") ||
         Name.startswith("aarch64.neon.bfm")) &&
        Name.endswith(".v4f32.v16i8")) {
      Intrinsic::ID IID =
          StringSwitch<Intrinsic::ID>(Name)
              .Case("arm.neon.bfmmla.v4f32.v16i8",
                    Intrinsic::arm_neon_bfmmla)
              .Case("arm.neon.bfmlalb.v4f32.v16i8",
                    Intrinsic::arm_neon_bfmlalb)
              .Case("arm.neon.bfmlalt.v4f32.v16i8",
                    Intrinsic::arm_neon_bfmlalt)
              .Case("aarch64.neon.bfmmla.v4f32.v16i8",
                    Intrinsic::aarch64_neon_bfmmla)
              .Case("aarch64.neon.bfmlalb.v4f32.v16i8",
                    Intrinsic::aarch64_neon_bfmlalb)
              .Case("aarch64.neon.bfmlalt.v4f32.v16i8",
                    Intrinsic::aarch64_neon_bfmlalt)
              .Default(Intrinsic::not_intrinsic);
      if (IID == Intrinsic::not_intrinsic)
        break;

      std::array<Type *, 0> Tys;
      NewFn = Intrinsic::getDeclaration(F->getParent(), IID, Tys);
      return true;
    }

    if (Name == "arm.mve.vctp64" &&
        cast<FixedVectorType>(F->getReturnType())->getNumElements() == 4) {
      // A vctp64 returning a v4i1 is converted to return a v2i1. Rename the
      // function and deal with it below in UpgradeIntrinsicCall.
      rename(F);
      return true;
    }
    // These too are changed to accept a v2i1 insteead of the old v4i1.
    if (Name == "arm.mve.mull.int.predicated.v2i64.v4i32.v4i1" ||
        Name == "arm.mve.vqdmull.predicated.v2i64.v4i32.v4i1" ||
        Name == "arm.mve.vldr.gather.base.predicated.v2i64.v2i64.v4i1" ||
        Name == "arm.mve.vldr.gather.base.wb.predicated.v2i64.v2i64.v4i1" ||
        Name ==
            "arm.mve.vldr.gather.offset.predicated.v2i64.p0i64.v2i64.v4i1" ||
        Name == "arm.mve.vldr.gather.offset.predicated.v2i64.p0.v2i64.v4i1" ||
        Name == "arm.mve.vstr.scatter.base.predicated.v2i64.v2i64.v4i1" ||
        Name == "arm.mve.vstr.scatter.base.wb.predicated.v2i64.v2i64.v4i1" ||
        Name ==
            "arm.mve.vstr.scatter.offset.predicated.p0i64.v2i64.v2i64.v4i1" ||
        Name == "arm.mve.vstr.scatter.offset.predicated.p0.v2i64.v2i64.v4i1" ||
        Name == "arm.cde.vcx1q.predicated.v2i64.v4i1" ||
        Name == "arm.cde.vcx1qa.predicated.v2i64.v4i1" ||
        Name == "arm.cde.vcx2q.predicated.v2i64.v4i1" ||
        Name == "arm.cde.vcx2qa.predicated.v2i64.v4i1" ||
        Name == "arm.cde.vcx3q.predicated.v2i64.v4i1" ||
        Name == "arm.cde.vcx3qa.predicated.v2i64.v4i1")
      return true;

    if (Name.startswith("amdgcn."))
      Name = Name.substr(7); // Strip off "amdgcn."

    if (Name == "alignbit") {
      // Target specific intrinsic became redundant
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::fshr,
                                        {F->getReturnType()});
      return true;
    }

    if (Name.startswith("atomic.inc") || Name.startswith("atomic.dec")) {
      // This was replaced with atomicrmw uinc_wrap and udec_wrap, so there's no
      // new declaration.
      NewFn = nullptr;
      return true;
    }

    break;
  }
  case 'c': {
    if (Name.startswith("ctlz.") && F->arg_size() == 1) {
      rename(F);
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::ctlz,
                                        F->arg_begin()->getType());
      return true;
    }
    if (Name.startswith("cttz.") && F->arg_size() == 1) {
      rename(F);
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::cttz,
                                        F->arg_begin()->getType());
      return true;
    }
    break;
  }
  case 'd': {
    if (Name == "dbg.addr") {
      rename(F);
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::dbg_value);
      return true;
    }
    if (Name == "dbg.value" && F->arg_size() == 4) {
      rename(F);
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::dbg_value);
      return true;
    }
    break;
  }
  case 'e': {
    if (Name.startswith("experimental.vector.extract.")) {
      rename(F);
      Type *Tys[] = {F->getReturnType(), F->arg_begin()->getType()};
      NewFn = Intrinsic::getDeclaration(F->getParent(),
                                        Intrinsic::vector_extract, Tys);
      return true;
    }

    if (Name.startswith("experimental.vector.insert.")) {
      rename(F);
      auto Args = F->getFunctionType()->params();
      Type *Tys[] = {Args[0], Args[1]};
      NewFn = Intrinsic::getDeclaration(F->getParent(),
                                        Intrinsic::vector_insert, Tys);
      return true;
    }

    SmallVector<StringRef, 2> Groups;
    static const Regex R("^experimental.vector.reduce.([a-z]+)\\.[a-z][0-9]+");
    if (R.match(Name, &Groups)) {
      Intrinsic::ID ID;
      ID = StringSwitch<Intrinsic::ID>(Groups[1])
               .Case("add", Intrinsic::vector_reduce_add)
               .Case("mul", Intrinsic::vector_reduce_mul)
               .Case("and", Intrinsic::vector_reduce_and)
               .Case("or", Intrinsic::vector_reduce_or)
               .Case("xor", Intrinsic::vector_reduce_xor)
               .Case("smax", Intrinsic::vector_reduce_smax)
               .Case("smin", Intrinsic::vector_reduce_smin)
               .Case("umax", Intrinsic::vector_reduce_umax)
               .Case("umin", Intrinsic::vector_reduce_umin)
               .Case("fmax", Intrinsic::vector_reduce_fmax)
               .Case("fmin", Intrinsic::vector_reduce_fmin)
               .Default(Intrinsic::not_intrinsic);
      if (ID != Intrinsic::not_intrinsic) {
        rename(F);
        auto Args = F->getFunctionType()->params();
        NewFn = Intrinsic::getDeclaration(F->getParent(), ID, {Args[0]});
        return true;
      }
    }
    static const Regex R2(
        "^experimental.vector.reduce.v2.([a-z]+)\\.[fi][0-9]+");
    Groups.clear();
    if (R2.match(Name, &Groups)) {
      Intrinsic::ID ID = Intrinsic::not_intrinsic;
      if (Groups[1] == "fadd")
        ID = Intrinsic::vector_reduce_fadd;
      if (Groups[1] == "fmul")
        ID = Intrinsic::vector_reduce_fmul;
      if (ID != Intrinsic::not_intrinsic) {
        rename(F);
        auto Args = F->getFunctionType()->params();
        Type *Tys[] = {Args[1]};
        NewFn = Intrinsic::getDeclaration(F->getParent(), ID, Tys);
        return true;
      }
    }
    break;
  }
  case 'f':
    if (Name.startswith("flt.rounds")) {
      rename(F);
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::get_rounding);
      return true;
    }
    break;
  case 'i':
  case 'l': {
    bool IsLifetimeStart = Name.startswith("lifetime.start");
    if (IsLifetimeStart || Name.startswith("invariant.start")) {
      Intrinsic::ID ID = IsLifetimeStart ?
        Intrinsic::lifetime_start : Intrinsic::invariant_start;
      auto Args = F->getFunctionType()->params();
      Type* ObjectPtr[1] = {Args[1]};
      if (F->getName() != Intrinsic::getName(ID, ObjectPtr, F->getParent())) {
        rename(F);
        NewFn = Intrinsic::getDeclaration(F->getParent(), ID, ObjectPtr);
        return true;
      }
    }

    bool IsLifetimeEnd = Name.startswith("lifetime.end");
    if (IsLifetimeEnd || Name.startswith("invariant.end")) {
      Intrinsic::ID ID = IsLifetimeEnd ?
        Intrinsic::lifetime_end : Intrinsic::invariant_end;

      auto Args = F->getFunctionType()->params();
      Type* ObjectPtr[1] = {Args[IsLifetimeEnd ? 1 : 2]};
      if (F->getName() != Intrinsic::getName(ID, ObjectPtr, F->getParent())) {
        rename(F);
        NewFn = Intrinsic::getDeclaration(F->getParent(), ID, ObjectPtr);
        return true;
      }
    }
    if (Name.startswith("invariant.group.barrier")) {
      // Rename invariant.group.barrier to launder.invariant.group
      auto Args = F->getFunctionType()->params();
      Type* ObjectPtr[1] = {Args[0]};
      rename(F);
      NewFn = Intrinsic::getDeclaration(F->getParent(),
          Intrinsic::launder_invariant_group, ObjectPtr);
      return true;

    }

    break;
  }
  case 'm': {
    if (Name.startswith("masked.load.")) {
      Type *Tys[] = { F->getReturnType(), F->arg_begin()->getType() };
      if (F->getName() !=
          Intrinsic::getName(Intrinsic::masked_load, Tys, F->getParent())) {
        rename(F);
        NewFn = Intrinsic::getDeclaration(F->getParent(),
                                          Intrinsic::masked_load,
                                          Tys);
        return true;
      }
    }
    if (Name.startswith("masked.store.")) {
      auto Args = F->getFunctionType()->params();
      Type *Tys[] = { Args[0], Args[1] };
      if (F->getName() !=
          Intrinsic::getName(Intrinsic::masked_store, Tys, F->getParent())) {
        rename(F);
        NewFn = Intrinsic::getDeclaration(F->getParent(),
                                          Intrinsic::masked_store,
                                          Tys);
        return true;
      }
    }
    // Renaming gather/scatter intrinsics with no address space overloading
    // to the new overload which includes an address space
    if (Name.startswith("masked.gather.")) {
      Type *Tys[] = {F->getReturnType(), F->arg_begin()->getType()};
      if (F->getName() !=
          Intrinsic::getName(Intrinsic::masked_gather, Tys, F->getParent())) {
        rename(F);
        NewFn = Intrinsic::getDeclaration(F->getParent(),
                                          Intrinsic::masked_gather, Tys);
        return true;
      }
    }
    if (Name.startswith("masked.scatter.")) {
      auto Args = F->getFunctionType()->params();
      Type *Tys[] = {Args[0], Args[1]};
      if (F->getName() !=
          Intrinsic::getName(Intrinsic::masked_scatter, Tys, F->getParent())) {
        rename(F);
        NewFn = Intrinsic::getDeclaration(F->getParent(),
                                          Intrinsic::masked_scatter, Tys);
        return true;
      }
    }
    // Updating the memory intrinsics (memcpy/memmove/memset) that have an
    // alignment parameter to embedding the alignment as an attribute of
    // the pointer args.
    if (Name.startswith("memcpy.") && F->arg_size() == 5) {
      rename(F);
      // Get the types of dest, src, and len
      ArrayRef<Type *> ParamTypes = F->getFunctionType()->params().slice(0, 3);
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::memcpy,
                                        ParamTypes);
      return true;
    }
    if (Name.startswith("memmove.") && F->arg_size() == 5) {
      rename(F);
      // Get the types of dest, src, and len
      ArrayRef<Type *> ParamTypes = F->getFunctionType()->params().slice(0, 3);
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::memmove,
                                        ParamTypes);
      return true;
    }
    if (Name.startswith("memset.") && F->arg_size() == 5) {
      rename(F);
      // Get the types of dest, and len
      const auto *FT = F->getFunctionType();
      Type *ParamTypes[2] = {
          FT->getParamType(0), // Dest
          FT->getParamType(2)  // len
      };
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::memset,
                                        ParamTypes);
      return true;
    }
    break;
  }
  case 'n': {
    if (Name.startswith("nvvm.")) {
      Name = Name.substr(5);

      // The following nvvm intrinsics correspond exactly to an LLVM intrinsic.
      Intrinsic::ID IID = StringSwitch<Intrinsic::ID>(Name)
                              .Cases("brev32", "brev64", Intrinsic::bitreverse)
                              .Case("clz.i", Intrinsic::ctlz)
                              .Case("popc.i", Intrinsic::ctpop)
                              .Default(Intrinsic::not_intrinsic);
      if (IID != Intrinsic::not_intrinsic && F->arg_size() == 1) {
        NewFn = Intrinsic::getDeclaration(F->getParent(), IID,
                                          {F->getReturnType()});
        return true;
      }
      IID = ShouldUpgradeNVPTXBF16Intrinsic(Name);
      if (IID != Intrinsic::not_intrinsic &&
          !F->getReturnType()->getScalarType()->isBFloatTy()) {
        NewFn = nullptr;
        return true;
      }
      // The following nvvm intrinsics correspond exactly to an LLVM idiom, but
      // not to an intrinsic alone.  We expand them in UpgradeIntrinsicCall.
      //
      // TODO: We could add lohi.i2d.
      bool Expand = StringSwitch<bool>(Name)
                        .Cases("abs.i", "abs.ll", true)
                        .Cases("clz.ll", "popc.ll", "h2f", true)
                        .Cases("max.i", "max.ll", "max.ui", "max.ull", true)
                        .Cases("min.i", "min.ll", "min.ui", "min.ull", true)
                        .StartsWith("atomic.load.add.f32.p", true)
                        .StartsWith("atomic.load.add.f64.p", true)
                        .Default(false);
      if (Expand) {
        NewFn = nullptr;
        return true;
      }
    }
    break;
  }
  case 'o':
    // We only need to change the name to match the mangling including the
    // address space.
    if (Name.startswith("objectsize.")) {
      Type *Tys[2] = { F->getReturnType(), F->arg_begin()->getType() };
      if (F->arg_size() == 2 || F->arg_size() == 3 ||
          F->getName() !=
              Intrinsic::getName(Intrinsic::objectsize, Tys, F->getParent())) {
        rename(F);
        NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::objectsize,
                                          Tys);
        return true;
      }
    }
    break;

  case 'p':
    if (Name == "prefetch") {
      // Handle address space overloading.
      Type *Tys[] = {F->arg_begin()->getType()};
      if (F->getName() !=
          Intrinsic::getName(Intrinsic::prefetch, Tys, F->getParent())) {
        rename(F);
        NewFn =
            Intrinsic::getDeclaration(F->getParent(), Intrinsic::prefetch, Tys);
        return true;
      }
    } else if (Name.startswith("ptr.annotation.") && F->arg_size() == 4) {
      rename(F);
      NewFn = Intrinsic::getDeclaration(
          F->getParent(), Intrinsic::ptr_annotation,
          {F->arg_begin()->getType(), F->getArg(1)->getType()});
      return true;
    }
    break;

  case 'r':
    if (Name == "riscv.aes32dsi" &&
        !F->getFunctionType()->getParamType(2)->isIntegerTy(32)) {
      rename(F);
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::riscv_aes32dsi);
      return true;
    }
    if (Name == "riscv.aes32dsmi" &&
        !F->getFunctionType()->getParamType(2)->isIntegerTy(32)) {
      rename(F);
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::riscv_aes32dsmi);
      return true;
    }
    if (Name == "riscv.aes32esi" &&
        !F->getFunctionType()->getParamType(2)->isIntegerTy(32)) {
      rename(F);
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::riscv_aes32esi);
      return true;
    }
    if (Name == "riscv.aes32esmi" &&
        !F->getFunctionType()->getParamType(2)->isIntegerTy(32)) {
      rename(F);
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::riscv_aes32esmi);
      return true;
    }
    if (Name.startswith("riscv.sm4ks") &&
        (!F->getFunctionType()->getParamType(2)->isIntegerTy(32) ||
         F->getFunctionType()->getReturnType()->isIntegerTy(64))) {
      rename(F);
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::riscv_sm4ks);
      return true;
    }
    if (Name.startswith("riscv.sm4ed") &&
        (!F->getFunctionType()->getParamType(2)->isIntegerTy(32) ||
         F->getFunctionType()->getReturnType()->isIntegerTy(64))) {
      rename(F);
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::riscv_sm4ed);
      return true;
    }
    if (Name.startswith("riscv.sha256sig0") &&
        F->getFunctionType()->getReturnType()->isIntegerTy(64)) {
      rename(F);
      NewFn = Intrinsic::getDeclaration(F->getParent(),
                                        Intrinsic::riscv_sha256sig0);
      return true;
    }
    if (Name.startswith("riscv.sha256sig1") &&
        F->getFunctionType()->getReturnType()->isIntegerTy(64)) {
      rename(F);
      NewFn = Intrinsic::getDeclaration(F->getParent(),
                                        Intrinsic::riscv_sha256sig1);
      return true;
    }
    if (Name.startswith("riscv.sha256sum0") &&
        F->getFunctionType()->getReturnType()->isIntegerTy(64)) {
      rename(F);
      NewFn = Intrinsic::getDeclaration(F->getParent(),
                                        Intrinsic::riscv_sha256sum0);
      return true;
    }
    if (Name.startswith("riscv.sha256sum1") &&
        F->getFunctionType()->getReturnType()->isIntegerTy(64)) {
      rename(F);
      NewFn = Intrinsic::getDeclaration(F->getParent(),
                                        Intrinsic::riscv_sha256sum1);
      return true;
    }
    if (Name.startswith("riscv.sm3p0") &&
        F->getFunctionType()->getReturnType()->isIntegerTy(64)) {
      rename(F);
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::riscv_sm3p0);
      return true;
    }
    if (Name.startswith("riscv.sm3p1") &&
        F->getFunctionType()->getReturnType()->isIntegerTy(64)) {
      rename(F);
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::riscv_sm3p1);
      return true;
    }
    break;

  case 's':
    if (Name == "stackprotectorcheck") {
      NewFn = nullptr;
      return true;
    }
    break;

  case 'v': {
    if (Name == "var.annotation" && F->arg_size() == 4) {
      rename(F);
      NewFn = Intrinsic::getDeclaration(
          F->getParent(), Intrinsic::var_annotation,
          {{F->arg_begin()->getType(), F->getArg(1)->getType()}});
      return true;
    }
    break;
  }

  case 'w':
    if (Name.startswith("wasm.fma.")) {
      rename(F);
      NewFn = Intrinsic::getDeclaration(
          F->getParent(), Intrinsic::wasm_relaxed_madd, F->getReturnType());
      return true;
    }
    if (Name.startswith("wasm.fms.")) {
      rename(F);
      NewFn = Intrinsic::getDeclaration(
          F->getParent(), Intrinsic::wasm_relaxed_nmadd, F->getReturnType());
      return true;
    }
    if (Name.startswith("wasm.laneselect.")) {
      rename(F);
      NewFn = Intrinsic::getDeclaration(
          F->getParent(), Intrinsic::wasm_relaxed_laneselect,
          F->getReturnType());
      return true;
    }
    if (Name == "wasm.dot.i8x16.i7x16.signed") {
      rename(F);
      NewFn = Intrinsic::getDeclaration(
          F->getParent(), Intrinsic::wasm_relaxed_dot_i8x16_i7x16_signed);
      return true;
    }
    if (Name == "wasm.dot.i8x16.i7x16.add.signed") {
      rename(F);
      NewFn = Intrinsic::getDeclaration(
          F->getParent(), Intrinsic::wasm_relaxed_dot_i8x16_i7x16_add_signed);
      return true;
    }
    break;

  case 'x':
    if (UpgradeX86IntrinsicFunction(F, Name, NewFn))
      return true;
  }

  auto *ST = dyn_cast<StructType>(F->getReturnType());
  if (ST && (!ST->isLiteral() || ST->isPacked())) {
    // Replace return type with literal non-packed struct. Only do this for
    // intrinsics declared to return a struct, not for intrinsics with
    // overloaded return type, in which case the exact struct type will be
    // mangled into the name.
    SmallVector<Intrinsic::IITDescriptor> Desc;
    Intrinsic::getIntrinsicInfoTableEntries(F->getIntrinsicID(), Desc);
    if (Desc.front().Kind == Intrinsic::IITDescriptor::Struct) {
      auto *FT = F->getFunctionType();
      auto *NewST = StructType::get(ST->getContext(), ST->elements());
      auto *NewFT = FunctionType::get(NewST, FT->params(), FT->isVarArg());
      std::string Name = F->getName().str();
      rename(F);
      NewFn = Function::Create(NewFT, F->getLinkage(), F->getAddressSpace(),
                               Name, F->getParent());

      // The new function may also need remangling.
      if (auto Result = llvm::Intrinsic::remangleIntrinsicFunction(NewFn))
        NewFn = *Result;
      return true;
    }
  }

  // Remangle our intrinsic since we upgrade the mangling
  auto Result = llvm::Intrinsic::remangleIntrinsicFunction(F);
  if (Result != std::nullopt) {
    NewFn = *Result;
    return true;
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

GlobalVariable *llvm::UpgradeGlobalVariable(GlobalVariable *GV) {
  if (!(GV->hasName() && (GV->getName() == "llvm.global_ctors" ||
                          GV->getName() == "llvm.global_dtors")) ||
      !GV->hasInitializer())
    return nullptr;
  ArrayType *ATy = dyn_cast<ArrayType>(GV->getValueType());
  if (!ATy)
    return nullptr;
  StructType *STy = dyn_cast<StructType>(ATy->getElementType());
  if (!STy || STy->getNumElements() != 2)
    return nullptr;

  LLVMContext &C = GV->getContext();
  IRBuilder<> IRB(C);
  auto EltTy = StructType::get(STy->getElementType(0), STy->getElementType(1),
                               IRB.getInt8PtrTy());
  Constant *Init = GV->getInitializer();
  unsigned N = Init->getNumOperands();
  std::vector<Constant *> NewCtors(N);
  for (unsigned i = 0; i != N; ++i) {
    auto Ctor = cast<Constant>(Init->getOperand(i));
    NewCtors[i] = ConstantStruct::get(
        EltTy, Ctor->getAggregateElement(0u), Ctor->getAggregateElement(1),
        Constant::getNullValue(IRB.getInt8PtrTy()));
  }
  Constant *NewInit = ConstantArray::get(ArrayType::get(EltTy, N), NewCtors);

  return new GlobalVariable(NewInit->getType(), false, GV->getLinkage(),
                            NewInit, GV->getName());
}

// Handles upgrading SSE2/AVX2/AVX512BW PSLLDQ intrinsics by converting them
// to byte shuffles.
static Value *UpgradeX86PSLLDQIntrinsics(IRBuilder<> &Builder,
                                         Value *Op, unsigned Shift) {
  auto *ResultTy = cast<FixedVectorType>(Op->getType());
  unsigned NumElts = ResultTy->getNumElements() * 8;

  // Bitcast from a 64-bit element type to a byte element type.
  Type *VecTy = FixedVectorType::get(Builder.getInt8Ty(), NumElts);
  Op = Builder.CreateBitCast(Op, VecTy, "cast");

  // We'll be shuffling in zeroes.
  Value *Res = Constant::getNullValue(VecTy);

  // If shift is less than 16, emit a shuffle to move the bytes. Otherwise,
  // we'll just return the zero vector.
  if (Shift < 16) {
    int Idxs[64];
    // 256/512-bit version is split into 2/4 16-byte lanes.
    for (unsigned l = 0; l != NumElts; l += 16)
      for (unsigned i = 0; i != 16; ++i) {
        unsigned Idx = NumElts + i - Shift;
        if (Idx < NumElts)
          Idx -= NumElts - 16; // end of lane, switch operand.
        Idxs[l + i] = Idx + l;
      }

    Res = Builder.CreateShuffleVector(Res, Op, ArrayRef(Idxs, NumElts));
  }

  // Bitcast back to a 64-bit element type.
  return Builder.CreateBitCast(Res, ResultTy, "cast");
}

// Handles upgrading SSE2/AVX2/AVX512BW PSRLDQ intrinsics by converting them
// to byte shuffles.
static Value *UpgradeX86PSRLDQIntrinsics(IRBuilder<> &Builder, Value *Op,
                                         unsigned Shift) {
  auto *ResultTy = cast<FixedVectorType>(Op->getType());
  unsigned NumElts = ResultTy->getNumElements() * 8;

  // Bitcast from a 64-bit element type to a byte element type.
  Type *VecTy = FixedVectorType::get(Builder.getInt8Ty(), NumElts);
  Op = Builder.CreateBitCast(Op, VecTy, "cast");

  // We'll be shuffling in zeroes.
  Value *Res = Constant::getNullValue(VecTy);

  // If shift is less than 16, emit a shuffle to move the bytes. Otherwise,
  // we'll just return the zero vector.
  if (Shift < 16) {
    int Idxs[64];
    // 256/512-bit version is split into 2/4 16-byte lanes.
    for (unsigned l = 0; l != NumElts; l += 16)
      for (unsigned i = 0; i != 16; ++i) {
        unsigned Idx = i + Shift;
        if (Idx >= 16)
          Idx += NumElts - 16; // end of lane, switch operand.
        Idxs[l + i] = Idx + l;
      }

    Res = Builder.CreateShuffleVector(Op, Res, ArrayRef(Idxs, NumElts));
  }

  // Bitcast back to a 64-bit element type.
  return Builder.CreateBitCast(Res, ResultTy, "cast");
}

static Value *getX86MaskVec(IRBuilder<> &Builder, Value *Mask,
                            unsigned NumElts) {
  assert(isPowerOf2_32(NumElts) && "Expected power-of-2 mask elements");
  llvm::VectorType *MaskTy = FixedVectorType::get(
      Builder.getInt1Ty(), cast<IntegerType>(Mask->getType())->getBitWidth());
  Mask = Builder.CreateBitCast(Mask, MaskTy);

  // If we have less than 8 elements (1, 2 or 4), then the starting mask was an
  // i8 and we need to extract down to the right number of elements.
  if (NumElts <= 4) {
    int Indices[4];
    for (unsigned i = 0; i != NumElts; ++i)
      Indices[i] = i;
    Mask = Builder.CreateShuffleVector(Mask, Mask, ArrayRef(Indices, NumElts),
                                       "extract");
  }

  return Mask;
}

static Value *EmitX86Select(IRBuilder<> &Builder, Value *Mask,
                            Value *Op0, Value *Op1) {
  // If the mask is all ones just emit the first operation.
  if (const auto *C = dyn_cast<Constant>(Mask))
    if (C->isAllOnesValue())
      return Op0;

  Mask = getX86MaskVec(Builder, Mask,
                       cast<FixedVectorType>(Op0->getType())->getNumElements());
  return Builder.CreateSelect(Mask, Op0, Op1);
}

static Value *EmitX86ScalarSelect(IRBuilder<> &Builder, Value *Mask,
                                  Value *Op0, Value *Op1) {
  // If the mask is all ones just emit the first operation.
  if (const auto *C = dyn_cast<Constant>(Mask))
    if (C->isAllOnesValue())
      return Op0;

  auto *MaskTy = FixedVectorType::get(Builder.getInt1Ty(),
                                      Mask->getType()->getIntegerBitWidth());
  Mask = Builder.CreateBitCast(Mask, MaskTy);
  Mask = Builder.CreateExtractElement(Mask, (uint64_t)0);
  return Builder.CreateSelect(Mask, Op0, Op1);
}

// Handle autoupgrade for masked PALIGNR and VALIGND/Q intrinsics.
// PALIGNR handles large immediates by shifting while VALIGN masks the immediate
// so we need to handle both cases. VALIGN also doesn't have 128-bit lanes.
static Value *UpgradeX86ALIGNIntrinsics(IRBuilder<> &Builder, Value *Op0,
                                        Value *Op1, Value *Shift,
                                        Value *Passthru, Value *Mask,
                                        bool IsVALIGN) {
  unsigned ShiftVal = cast<llvm::ConstantInt>(Shift)->getZExtValue();

  unsigned NumElts = cast<FixedVectorType>(Op0->getType())->getNumElements();
  assert((IsVALIGN || NumElts % 16 == 0) && "Illegal NumElts for PALIGNR!");
  assert((!IsVALIGN || NumElts <= 16) && "NumElts too large for VALIGN!");
  assert(isPowerOf2_32(NumElts) && "NumElts not a power of 2!");

  // Mask the immediate for VALIGN.
  if (IsVALIGN)
    ShiftVal &= (NumElts - 1);

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

  int Indices[64];
  // 256-bit palignr operates on 128-bit lanes so we need to handle that
  for (unsigned l = 0; l < NumElts; l += 16) {
    for (unsigned i = 0; i != 16; ++i) {
      unsigned Idx = ShiftVal + i;
      if (!IsVALIGN && Idx >= 16) // Disable wrap for VALIGN.
        Idx += NumElts - 16; // End of lane, switch operand.
      Indices[l + i] = Idx + l;
    }
  }

  Value *Align = Builder.CreateShuffleVector(
      Op1, Op0, ArrayRef(Indices, NumElts), "palignr");

  return EmitX86Select(Builder, Mask, Align, Passthru);
}

static Value *UpgradeX86VPERMT2Intrinsics(IRBuilder<> &Builder, CallBase &CI,
                                          bool ZeroMask, bool IndexForm) {
  Type *Ty = CI.getType();
  unsigned VecWidth = Ty->getPrimitiveSizeInBits();
  unsigned EltWidth = Ty->getScalarSizeInBits();
  bool IsFloat = Ty->isFPOrFPVectorTy();
  Intrinsic::ID IID;
  if (VecWidth == 128 && EltWidth == 32 && IsFloat)
    IID = Intrinsic::x86_avx512_vpermi2var_ps_128;
  else if (VecWidth == 128 && EltWidth == 32 && !IsFloat)
    IID = Intrinsic::x86_avx512_vpermi2var_d_128;
  else if (VecWidth == 128 && EltWidth == 64 && IsFloat)
    IID = Intrinsic::x86_avx512_vpermi2var_pd_128;
  else if (VecWidth == 128 && EltWidth == 64 && !IsFloat)
    IID = Intrinsic::x86_avx512_vpermi2var_q_128;
  else if (VecWidth == 256 && EltWidth == 32 && IsFloat)
    IID = Intrinsic::x86_avx512_vpermi2var_ps_256;
  else if (VecWidth == 256 && EltWidth == 32 && !IsFloat)
    IID = Intrinsic::x86_avx512_vpermi2var_d_256;
  else if (VecWidth == 256 && EltWidth == 64 && IsFloat)
    IID = Intrinsic::x86_avx512_vpermi2var_pd_256;
  else if (VecWidth == 256 && EltWidth == 64 && !IsFloat)
    IID = Intrinsic::x86_avx512_vpermi2var_q_256;
  else if (VecWidth == 512 && EltWidth == 32 && IsFloat)
    IID = Intrinsic::x86_avx512_vpermi2var_ps_512;
  else if (VecWidth == 512 && EltWidth == 32 && !IsFloat)
    IID = Intrinsic::x86_avx512_vpermi2var_d_512;
  else if (VecWidth == 512 && EltWidth == 64 && IsFloat)
    IID = Intrinsic::x86_avx512_vpermi2var_pd_512;
  else if (VecWidth == 512 && EltWidth == 64 && !IsFloat)
    IID = Intrinsic::x86_avx512_vpermi2var_q_512;
  else if (VecWidth == 128 && EltWidth == 16)
    IID = Intrinsic::x86_avx512_vpermi2var_hi_128;
  else if (VecWidth == 256 && EltWidth == 16)
    IID = Intrinsic::x86_avx512_vpermi2var_hi_256;
  else if (VecWidth == 512 && EltWidth == 16)
    IID = Intrinsic::x86_avx512_vpermi2var_hi_512;
  else if (VecWidth == 128 && EltWidth == 8)
    IID = Intrinsic::x86_avx512_vpermi2var_qi_128;
  else if (VecWidth == 256 && EltWidth == 8)
    IID = Intrinsic::x86_avx512_vpermi2var_qi_256;
  else if (VecWidth == 512 && EltWidth == 8)
    IID = Intrinsic::x86_avx512_vpermi2var_qi_512;
  else
    llvm_unreachable("Unexpected intrinsic");

  Value *Args[] = { CI.getArgOperand(0) , CI.getArgOperand(1),
                    CI.getArgOperand(2) };

  // If this isn't index form we need to swap operand 0 and 1.
  if (!IndexForm)
    std::swap(Args[0], Args[1]);

  Value *V = Builder.CreateCall(Intrinsic::getDeclaration(CI.getModule(), IID),
                                Args);
  Value *PassThru = ZeroMask ? ConstantAggregateZero::get(Ty)
                             : Builder.CreateBitCast(CI.getArgOperand(1),
                                                     Ty);
  return EmitX86Select(Builder, CI.getArgOperand(3), V, PassThru);
}

static Value *UpgradeX86BinaryIntrinsics(IRBuilder<> &Builder, CallBase &CI,
                                         Intrinsic::ID IID) {
  Type *Ty = CI.getType();
  Value *Op0 = CI.getOperand(0);
  Value *Op1 = CI.getOperand(1);
  Function *Intrin = Intrinsic::getDeclaration(CI.getModule(), IID, Ty);
  Value *Res = Builder.CreateCall(Intrin, {Op0, Op1});

  if (CI.arg_size() == 4) { // For masked intrinsics.
    Value *VecSrc = CI.getOperand(2);
    Value *Mask = CI.getOperand(3);
    Res = EmitX86Select(Builder, Mask, Res, VecSrc);
  }
  return Res;
}

static Value *upgradeX86Rotate(IRBuilder<> &Builder, CallBase &CI,
                               bool IsRotateRight) {
  Type *Ty = CI.getType();
  Value *Src = CI.getArgOperand(0);
  Value *Amt = CI.getArgOperand(1);

  // Amount may be scalar immediate, in which case create a splat vector.
  // Funnel shifts amounts are treated as modulo and types are all power-of-2 so
  // we only care about the lowest log2 bits anyway.
  if (Amt->getType() != Ty) {
    unsigned NumElts = cast<FixedVectorType>(Ty)->getNumElements();
    Amt = Builder.CreateIntCast(Amt, Ty->getScalarType(), false);
    Amt = Builder.CreateVectorSplat(NumElts, Amt);
  }

  Intrinsic::ID IID = IsRotateRight ? Intrinsic::fshr : Intrinsic::fshl;
  Function *Intrin = Intrinsic::getDeclaration(CI.getModule(), IID, Ty);
  Value *Res = Builder.CreateCall(Intrin, {Src, Src, Amt});

  if (CI.arg_size() == 4) { // For masked intrinsics.
    Value *VecSrc = CI.getOperand(2);
    Value *Mask = CI.getOperand(3);
    Res = EmitX86Select(Builder, Mask, Res, VecSrc);
  }
  return Res;
}

static Value *upgradeX86vpcom(IRBuilder<> &Builder, CallBase &CI, unsigned Imm,
                              bool IsSigned) {
  Type *Ty = CI.getType();
  Value *LHS = CI.getArgOperand(0);
  Value *RHS = CI.getArgOperand(1);

  CmpInst::Predicate Pred;
  switch (Imm) {
  case 0x0:
    Pred = IsSigned ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT;
    break;
  case 0x1:
    Pred = IsSigned ? ICmpInst::ICMP_SLE : ICmpInst::ICMP_ULE;
    break;
  case 0x2:
    Pred = IsSigned ? ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT;
    break;
  case 0x3:
    Pred = IsSigned ? ICmpInst::ICMP_SGE : ICmpInst::ICMP_UGE;
    break;
  case 0x4:
    Pred = ICmpInst::ICMP_EQ;
    break;
  case 0x5:
    Pred = ICmpInst::ICMP_NE;
    break;
  case 0x6:
    return Constant::getNullValue(Ty); // FALSE
  case 0x7:
    return Constant::getAllOnesValue(Ty); // TRUE
  default:
    llvm_unreachable("Unknown XOP vpcom/vpcomu predicate");
  }

  Value *Cmp = Builder.CreateICmp(Pred, LHS, RHS);
  Value *Ext = Builder.CreateSExt(Cmp, Ty);
  return Ext;
}

static Value *upgradeX86ConcatShift(IRBuilder<> &Builder, CallBase &CI,
                                    bool IsShiftRight, bool ZeroMask) {
  Type *Ty = CI.getType();
  Value *Op0 = CI.getArgOperand(0);
  Value *Op1 = CI.getArgOperand(1);
  Value *Amt = CI.getArgOperand(2);

  if (IsShiftRight)
    std::swap(Op0, Op1);

  // Amount may be scalar immediate, in which case create a splat vector.
  // Funnel shifts amounts are treated as modulo and types are all power-of-2 so
  // we only care about the lowest log2 bits anyway.
  if (Amt->getType() != Ty) {
    unsigned NumElts = cast<FixedVectorType>(Ty)->getNumElements();
    Amt = Builder.CreateIntCast(Amt, Ty->getScalarType(), false);
    Amt = Builder.CreateVectorSplat(NumElts, Amt);
  }

  Intrinsic::ID IID = IsShiftRight ? Intrinsic::fshr : Intrinsic::fshl;
  Function *Intrin = Intrinsic::getDeclaration(CI.getModule(), IID, Ty);
  Value *Res = Builder.CreateCall(Intrin, {Op0, Op1, Amt});

  unsigned NumArgs = CI.arg_size();
  if (NumArgs >= 4) { // For masked intrinsics.
    Value *VecSrc = NumArgs == 5 ? CI.getArgOperand(3) :
                    ZeroMask     ? ConstantAggregateZero::get(CI.getType()) :
                                   CI.getArgOperand(0);
    Value *Mask = CI.getOperand(NumArgs - 1);
    Res = EmitX86Select(Builder, Mask, Res, VecSrc);
  }
  return Res;
}

static Value *UpgradeMaskedStore(IRBuilder<> &Builder,
                                 Value *Ptr, Value *Data, Value *Mask,
                                 bool Aligned) {
  // Cast the pointer to the right type.
  Ptr = Builder.CreateBitCast(Ptr,
                              llvm::PointerType::getUnqual(Data->getType()));
  const Align Alignment =
      Aligned
          ? Align(Data->getType()->getPrimitiveSizeInBits().getFixedValue() / 8)
          : Align(1);

  // If the mask is all ones just emit a regular store.
  if (const auto *C = dyn_cast<Constant>(Mask))
    if (C->isAllOnesValue())
      return Builder.CreateAlignedStore(Data, Ptr, Alignment);

  // Convert the mask from an integer type to a vector of i1.
  unsigned NumElts = cast<FixedVectorType>(Data->getType())->getNumElements();
  Mask = getX86MaskVec(Builder, Mask, NumElts);
  return Builder.CreateMaskedStore(Data, Ptr, Alignment, Mask);
}

static Value *UpgradeMaskedLoad(IRBuilder<> &Builder,
                                Value *Ptr, Value *Passthru, Value *Mask,
                                bool Aligned) {
  Type *ValTy = Passthru->getType();
  // Cast the pointer to the right type.
  Ptr = Builder.CreateBitCast(Ptr, llvm::PointerType::getUnqual(ValTy));
  const Align Alignment =
      Aligned
          ? Align(
                Passthru->getType()->getPrimitiveSizeInBits().getFixedValue() /
                8)
          : Align(1);

  // If the mask is all ones just emit a regular store.
  if (const auto *C = dyn_cast<Constant>(Mask))
    if (C->isAllOnesValue())
      return Builder.CreateAlignedLoad(ValTy, Ptr, Alignment);

  // Convert the mask from an integer type to a vector of i1.
  unsigned NumElts = cast<FixedVectorType>(ValTy)->getNumElements();
  Mask = getX86MaskVec(Builder, Mask, NumElts);
  return Builder.CreateMaskedLoad(ValTy, Ptr, Alignment, Mask, Passthru);
}

static Value *upgradeAbs(IRBuilder<> &Builder, CallBase &CI) {
  Type *Ty = CI.getType();
  Value *Op0 = CI.getArgOperand(0);
  Function *F = Intrinsic::getDeclaration(CI.getModule(), Intrinsic::abs, Ty);
  Value *Res = Builder.CreateCall(F, {Op0, Builder.getInt1(false)});
  if (CI.arg_size() == 3)
    Res = EmitX86Select(Builder, CI.getArgOperand(2), Res, CI.getArgOperand(1));
  return Res;
}

static Value *upgradePMULDQ(IRBuilder<> &Builder, CallBase &CI, bool IsSigned) {
  Type *Ty = CI.getType();

  // Arguments have a vXi32 type so cast to vXi64.
  Value *LHS = Builder.CreateBitCast(CI.getArgOperand(0), Ty);
  Value *RHS = Builder.CreateBitCast(CI.getArgOperand(1), Ty);

  if (IsSigned) {
    // Shift left then arithmetic shift right.
    Constant *ShiftAmt = ConstantInt::get(Ty, 32);
    LHS = Builder.CreateShl(LHS, ShiftAmt);
    LHS = Builder.CreateAShr(LHS, ShiftAmt);
    RHS = Builder.CreateShl(RHS, ShiftAmt);
    RHS = Builder.CreateAShr(RHS, ShiftAmt);
  } else {
    // Clear the upper bits.
    Constant *Mask = ConstantInt::get(Ty, 0xffffffff);
    LHS = Builder.CreateAnd(LHS, Mask);
    RHS = Builder.CreateAnd(RHS, Mask);
  }

  Value *Res = Builder.CreateMul(LHS, RHS);

  if (CI.arg_size() == 4)
    Res = EmitX86Select(Builder, CI.getArgOperand(3), Res, CI.getArgOperand(2));

  return Res;
}

// Applying mask on vector of i1's and make sure result is at least 8 bits wide.
static Value *ApplyX86MaskOn1BitsVec(IRBuilder<> &Builder, Value *Vec,
                                     Value *Mask) {
  unsigned NumElts = cast<FixedVectorType>(Vec->getType())->getNumElements();
  if (Mask) {
    const auto *C = dyn_cast<Constant>(Mask);
    if (!C || !C->isAllOnesValue())
      Vec = Builder.CreateAnd(Vec, getX86MaskVec(Builder, Mask, NumElts));
  }

  if (NumElts < 8) {
    int Indices[8];
    for (unsigned i = 0; i != NumElts; ++i)
      Indices[i] = i;
    for (unsigned i = NumElts; i != 8; ++i)
      Indices[i] = NumElts + i % NumElts;
    Vec = Builder.CreateShuffleVector(Vec,
                                      Constant::getNullValue(Vec->getType()),
                                      Indices);
  }
  return Builder.CreateBitCast(Vec, Builder.getIntNTy(std::max(NumElts, 8U)));
}

static Value *upgradeMaskedCompare(IRBuilder<> &Builder, CallBase &CI,
                                   unsigned CC, bool Signed) {
  Value *Op0 = CI.getArgOperand(0);
  unsigned NumElts = cast<FixedVectorType>(Op0->getType())->getNumElements();

  Value *Cmp;
  if (CC == 3) {
    Cmp = Constant::getNullValue(
        FixedVectorType::get(Builder.getInt1Ty(), NumElts));
  } else if (CC == 7) {
    Cmp = Constant::getAllOnesValue(
        FixedVectorType::get(Builder.getInt1Ty(), NumElts));
  } else {
    ICmpInst::Predicate Pred;
    switch (CC) {
    default: llvm_unreachable("Unknown condition code");
    case 0: Pred = ICmpInst::ICMP_EQ;  break;
    case 1: Pred = Signed ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT; break;
    case 2: Pred = Signed ? ICmpInst::ICMP_SLE : ICmpInst::ICMP_ULE; break;
    case 4: Pred = ICmpInst::ICMP_NE;  break;
    case 5: Pred = Signed ? ICmpInst::ICMP_SGE : ICmpInst::ICMP_UGE; break;
    case 6: Pred = Signed ? ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT; break;
    }
    Cmp = Builder.CreateICmp(Pred, Op0, CI.getArgOperand(1));
  }

  Value *Mask = CI.getArgOperand(CI.arg_size() - 1);

  return ApplyX86MaskOn1BitsVec(Builder, Cmp, Mask);
}

// Replace a masked intrinsic with an older unmasked intrinsic.
static Value *UpgradeX86MaskedShift(IRBuilder<> &Builder, CallBase &CI,
                                    Intrinsic::ID IID) {
  Function *Intrin = Intrinsic::getDeclaration(CI.getModule(), IID);
  Value *Rep = Builder.CreateCall(Intrin,
                                 { CI.getArgOperand(0), CI.getArgOperand(1) });
  return EmitX86Select(Builder, CI.getArgOperand(3), Rep, CI.getArgOperand(2));
}

static Value* upgradeMaskedMove(IRBuilder<> &Builder, CallBase &CI) {
  Value* A = CI.getArgOperand(0);
  Value* B = CI.getArgOperand(1);
  Value* Src = CI.getArgOperand(2);
  Value* Mask = CI.getArgOperand(3);

  Value* AndNode = Builder.CreateAnd(Mask, APInt(8, 1));
  Value* Cmp = Builder.CreateIsNotNull(AndNode);
  Value* Extract1 = Builder.CreateExtractElement(B, (uint64_t)0);
  Value* Extract2 = Builder.CreateExtractElement(Src, (uint64_t)0);
  Value* Select = Builder.CreateSelect(Cmp, Extract1, Extract2);
  return Builder.CreateInsertElement(A, Select, (uint64_t)0);
}


static Value* UpgradeMaskToInt(IRBuilder<> &Builder, CallBase &CI) {
  Value* Op = CI.getArgOperand(0);
  Type* ReturnOp = CI.getType();
  unsigned NumElts = cast<FixedVectorType>(CI.getType())->getNumElements();
  Value *Mask = getX86MaskVec(Builder, Op, NumElts);
  return Builder.CreateSExt(Mask, ReturnOp, "vpmovm2");
}

// Replace intrinsic with unmasked version and a select.
static bool upgradeAVX512MaskToSelect(StringRef Name, IRBuilder<> &Builder,
                                      CallBase &CI, Value *&Rep) {
  Name = Name.substr(12); // Remove avx512.mask.

  unsigned VecWidth = CI.getType()->getPrimitiveSizeInBits();
  unsigned EltWidth = CI.getType()->getScalarSizeInBits();
  Intrinsic::ID IID;
  if (Name.startswith("max.p")) {
    if (VecWidth == 128 && EltWidth == 32)
      IID = Intrinsic::x86_sse_max_ps;
    else if (VecWidth == 128 && EltWidth == 64)
      IID = Intrinsic::x86_sse2_max_pd;
    else if (VecWidth == 256 && EltWidth == 32)
      IID = Intrinsic::x86_avx_max_ps_256;
    else if (VecWidth == 256 && EltWidth == 64)
      IID = Intrinsic::x86_avx_max_pd_256;
    else
      llvm_unreachable("Unexpected intrinsic");
  } else if (Name.startswith("min.p")) {
    if (VecWidth == 128 && EltWidth == 32)
      IID = Intrinsic::x86_sse_min_ps;
    else if (VecWidth == 128 && EltWidth == 64)
      IID = Intrinsic::x86_sse2_min_pd;
    else if (VecWidth == 256 && EltWidth == 32)
      IID = Intrinsic::x86_avx_min_ps_256;
    else if (VecWidth == 256 && EltWidth == 64)
      IID = Intrinsic::x86_avx_min_pd_256;
    else
      llvm_unreachable("Unexpected intrinsic");
  } else if (Name.startswith("pshuf.b.")) {
    if (VecWidth == 128)
      IID = Intrinsic::x86_ssse3_pshuf_b_128;
    else if (VecWidth == 256)
      IID = Intrinsic::x86_avx2_pshuf_b;
    else if (VecWidth == 512)
      IID = Intrinsic::x86_avx512_pshuf_b_512;
    else
      llvm_unreachable("Unexpected intrinsic");
  } else if (Name.startswith("pmul.hr.sw.")) {
    if (VecWidth == 128)
      IID = Intrinsic::x86_ssse3_pmul_hr_sw_128;
    else if (VecWidth == 256)
      IID = Intrinsic::x86_avx2_pmul_hr_sw;
    else if (VecWidth == 512)
      IID = Intrinsic::x86_avx512_pmul_hr_sw_512;
    else
      llvm_unreachable("Unexpected intrinsic");
  } else if (Name.startswith("pmulh.w.")) {
    if (VecWidth == 128)
      IID = Intrinsic::x86_sse2_pmulh_w;
    else if (VecWidth == 256)
      IID = Intrinsic::x86_avx2_pmulh_w;
    else if (VecWidth == 512)
      IID = Intrinsic::x86_avx512_pmulh_w_512;
    else
      llvm_unreachable("Unexpected intrinsic");
  } else if (Name.startswith("pmulhu.w.")) {
    if (VecWidth == 128)
      IID = Intrinsic::x86_sse2_pmulhu_w;
    else if (VecWidth == 256)
      IID = Intrinsic::x86_avx2_pmulhu_w;
    else if (VecWidth == 512)
      IID = Intrinsic::x86_avx512_pmulhu_w_512;
    else
      llvm_unreachable("Unexpected intrinsic");
  } else if (Name.startswith("pmaddw.d.")) {
    if (VecWidth == 128)
      IID = Intrinsic::x86_sse2_pmadd_wd;
    else if (VecWidth == 256)
      IID = Intrinsic::x86_avx2_pmadd_wd;
    else if (VecWidth == 512)
      IID = Intrinsic::x86_avx512_pmaddw_d_512;
    else
      llvm_unreachable("Unexpected intrinsic");
  } else if (Name.startswith("pmaddubs.w.")) {
    if (VecWidth == 128)
      IID = Intrinsic::x86_ssse3_pmadd_ub_sw_128;
    else if (VecWidth == 256)
      IID = Intrinsic::x86_avx2_pmadd_ub_sw;
    else if (VecWidth == 512)
      IID = Intrinsic::x86_avx512_pmaddubs_w_512;
    else
      llvm_unreachable("Unexpected intrinsic");
  } else if (Name.startswith("packsswb.")) {
    if (VecWidth == 128)
      IID = Intrinsic::x86_sse2_packsswb_128;
    else if (VecWidth == 256)
      IID = Intrinsic::x86_avx2_packsswb;
    else if (VecWidth == 512)
      IID = Intrinsic::x86_avx512_packsswb_512;
    else
      llvm_unreachable("Unexpected intrinsic");
  } else if (Name.startswith("packssdw.")) {
    if (VecWidth == 128)
      IID = Intrinsic::x86_sse2_packssdw_128;
    else if (VecWidth == 256)
      IID = Intrinsic::x86_avx2_packssdw;
    else if (VecWidth == 512)
      IID = Intrinsic::x86_avx512_packssdw_512;
    else
      llvm_unreachable("Unexpected intrinsic");
  } else if (Name.startswith("packuswb.")) {
    if (VecWidth == 128)
      IID = Intrinsic::x86_sse2_packuswb_128;
    else if (VecWidth == 256)
      IID = Intrinsic::x86_avx2_packuswb;
    else if (VecWidth == 512)
      IID = Intrinsic::x86_avx512_packuswb_512;
    else
      llvm_unreachable("Unexpected intrinsic");
  } else if (Name.startswith("packusdw.")) {
    if (VecWidth == 128)
      IID = Intrinsic::x86_sse41_packusdw;
    else if (VecWidth == 256)
      IID = Intrinsic::x86_avx2_packusdw;
    else if (VecWidth == 512)
      IID = Intrinsic::x86_avx512_packusdw_512;
    else
      llvm_unreachable("Unexpected intrinsic");
  } else if (Name.startswith("vpermilvar.")) {
    if (VecWidth == 128 && EltWidth == 32)
      IID = Intrinsic::x86_avx_vpermilvar_ps;
    else if (VecWidth == 128 && EltWidth == 64)
      IID = Intrinsic::x86_avx_vpermilvar_pd;
    else if (VecWidth == 256 && EltWidth == 32)
      IID = Intrinsic::x86_avx_vpermilvar_ps_256;
    else if (VecWidth == 256 && EltWidth == 64)
      IID = Intrinsic::x86_avx_vpermilvar_pd_256;
    else if (VecWidth == 512 && EltWidth == 32)
      IID = Intrinsic::x86_avx512_vpermilvar_ps_512;
    else if (VecWidth == 512 && EltWidth == 64)
      IID = Intrinsic::x86_avx512_vpermilvar_pd_512;
    else
      llvm_unreachable("Unexpected intrinsic");
  } else if (Name == "cvtpd2dq.256") {
    IID = Intrinsic::x86_avx_cvt_pd2dq_256;
  } else if (Name == "cvtpd2ps.256") {
    IID = Intrinsic::x86_avx_cvt_pd2_ps_256;
  } else if (Name == "cvttpd2dq.256") {
    IID = Intrinsic::x86_avx_cvtt_pd2dq_256;
  } else if (Name == "cvttps2dq.128") {
    IID = Intrinsic::x86_sse2_cvttps2dq;
  } else if (Name == "cvttps2dq.256") {
    IID = Intrinsic::x86_avx_cvtt_ps2dq_256;
  } else if (Name.startswith("permvar.")) {
    bool IsFloat = CI.getType()->isFPOrFPVectorTy();
    if (VecWidth == 256 && EltWidth == 32 && IsFloat)
      IID = Intrinsic::x86_avx2_permps;
    else if (VecWidth == 256 && EltWidth == 32 && !IsFloat)
      IID = Intrinsic::x86_avx2_permd;
    else if (VecWidth == 256 && EltWidth == 64 && IsFloat)
      IID = Intrinsic::x86_avx512_permvar_df_256;
    else if (VecWidth == 256 && EltWidth == 64 && !IsFloat)
      IID = Intrinsic::x86_avx512_permvar_di_256;
    else if (VecWidth == 512 && EltWidth == 32 && IsFloat)
      IID = Intrinsic::x86_avx512_permvar_sf_512;
    else if (VecWidth == 512 && EltWidth == 32 && !IsFloat)
      IID = Intrinsic::x86_avx512_permvar_si_512;
    else if (VecWidth == 512 && EltWidth == 64 && IsFloat)
      IID = Intrinsic::x86_avx512_permvar_df_512;
    else if (VecWidth == 512 && EltWidth == 64 && !IsFloat)
      IID = Intrinsic::x86_avx512_permvar_di_512;
    else if (VecWidth == 128 && EltWidth == 16)
      IID = Intrinsic::x86_avx512_permvar_hi_128;
    else if (VecWidth == 256 && EltWidth == 16)
      IID = Intrinsic::x86_avx512_permvar_hi_256;
    else if (VecWidth == 512 && EltWidth == 16)
      IID = Intrinsic::x86_avx512_permvar_hi_512;
    else if (VecWidth == 128 && EltWidth == 8)
      IID = Intrinsic::x86_avx512_permvar_qi_128;
    else if (VecWidth == 256 && EltWidth == 8)
      IID = Intrinsic::x86_avx512_permvar_qi_256;
    else if (VecWidth == 512 && EltWidth == 8)
      IID = Intrinsic::x86_avx512_permvar_qi_512;
    else
      llvm_unreachable("Unexpected intrinsic");
  } else if (Name.startswith("dbpsadbw.")) {
    if (VecWidth == 128)
      IID = Intrinsic::x86_avx512_dbpsadbw_128;
    else if (VecWidth == 256)
      IID = Intrinsic::x86_avx512_dbpsadbw_256;
    else if (VecWidth == 512)
      IID = Intrinsic::x86_avx512_dbpsadbw_512;
    else
      llvm_unreachable("Unexpected intrinsic");
  } else if (Name.startswith("pmultishift.qb.")) {
    if (VecWidth == 128)
      IID = Intrinsic::x86_avx512_pmultishift_qb_128;
    else if (VecWidth == 256)
      IID = Intrinsic::x86_avx512_pmultishift_qb_256;
    else if (VecWidth == 512)
      IID = Intrinsic::x86_avx512_pmultishift_qb_512;
    else
      llvm_unreachable("Unexpected intrinsic");
  } else if (Name.startswith("conflict.")) {
    if (Name[9] == 'd' && VecWidth == 128)
      IID = Intrinsic::x86_avx512_conflict_d_128;
    else if (Name[9] == 'd' && VecWidth == 256)
      IID = Intrinsic::x86_avx512_conflict_d_256;
    else if (Name[9] == 'd' && VecWidth == 512)
      IID = Intrinsic::x86_avx512_conflict_d_512;
    else if (Name[9] == 'q' && VecWidth == 128)
      IID = Intrinsic::x86_avx512_conflict_q_128;
    else if (Name[9] == 'q' && VecWidth == 256)
      IID = Intrinsic::x86_avx512_conflict_q_256;
    else if (Name[9] == 'q' && VecWidth == 512)
      IID = Intrinsic::x86_avx512_conflict_q_512;
    else
      llvm_unreachable("Unexpected intrinsic");
  } else if (Name.startswith("pavg.")) {
    if (Name[5] == 'b' && VecWidth == 128)
      IID = Intrinsic::x86_sse2_pavg_b;
    else if (Name[5] == 'b' && VecWidth == 256)
      IID = Intrinsic::x86_avx2_pavg_b;
    else if (Name[5] == 'b' && VecWidth == 512)
      IID = Intrinsic::x86_avx512_pavg_b_512;
    else if (Name[5] == 'w' && VecWidth == 128)
      IID = Intrinsic::x86_sse2_pavg_w;
    else if (Name[5] == 'w' && VecWidth == 256)
      IID = Intrinsic::x86_avx2_pavg_w;
    else if (Name[5] == 'w' && VecWidth == 512)
      IID = Intrinsic::x86_avx512_pavg_w_512;
    else
      llvm_unreachable("Unexpected intrinsic");
  } else
    return false;

  SmallVector<Value *, 4> Args(CI.args());
  Args.pop_back();
  Args.pop_back();
  Rep = Builder.CreateCall(Intrinsic::getDeclaration(CI.getModule(), IID),
                           Args);
  unsigned NumArgs = CI.arg_size();
  Rep = EmitX86Select(Builder, CI.getArgOperand(NumArgs - 1), Rep,
                      CI.getArgOperand(NumArgs - 2));
  return true;
}

/// Upgrade comment in call to inline asm that represents an objc retain release
/// marker.
void llvm::UpgradeInlineAsmString(std::string *AsmStr) {
  size_t Pos;
  if (AsmStr->find("mov\tfp") == 0 &&
      AsmStr->find("objc_retainAutoreleaseReturnValue") != std::string::npos &&
      (Pos = AsmStr->find("# marker")) != std::string::npos) {
    AsmStr->replace(Pos, 1, ";");
  }
}

static Value *UpgradeARMIntrinsicCall(StringRef Name, CallBase *CI, Function *F,
                                      IRBuilder<> &Builder) {
  if (Name == "mve.vctp64.old") {
    // Replace the old v4i1 vctp64 with a v2i1 vctp and predicate-casts to the
    // correct type.
    Value *VCTP = Builder.CreateCall(
        Intrinsic::getDeclaration(F->getParent(), Intrinsic::arm_mve_vctp64),
        CI->getArgOperand(0), CI->getName());
    Value *C1 = Builder.CreateCall(
        Intrinsic::getDeclaration(
            F->getParent(), Intrinsic::arm_mve_pred_v2i,
            {VectorType::get(Builder.getInt1Ty(), 2, false)}),
        VCTP);
    return Builder.CreateCall(
        Intrinsic::getDeclaration(
            F->getParent(), Intrinsic::arm_mve_pred_i2v,
            {VectorType::get(Builder.getInt1Ty(), 4, false)}),
        C1);
  } else if (Name == "mve.mull.int.predicated.v2i64.v4i32.v4i1" ||
             Name == "mve.vqdmull.predicated.v2i64.v4i32.v4i1" ||
             Name == "mve.vldr.gather.base.predicated.v2i64.v2i64.v4i1" ||
             Name == "mve.vldr.gather.base.wb.predicated.v2i64.v2i64.v4i1" ||
             Name ==
                 "mve.vldr.gather.offset.predicated.v2i64.p0i64.v2i64.v4i1" ||
             Name == "mve.vldr.gather.offset.predicated.v2i64.p0.v2i64.v4i1" ||
             Name == "mve.vstr.scatter.base.predicated.v2i64.v2i64.v4i1" ||
             Name == "mve.vstr.scatter.base.wb.predicated.v2i64.v2i64.v4i1" ||
             Name ==
                 "mve.vstr.scatter.offset.predicated.p0i64.v2i64.v2i64.v4i1" ||
             Name == "mve.vstr.scatter.offset.predicated.p0.v2i64.v2i64.v4i1" ||
             Name == "cde.vcx1q.predicated.v2i64.v4i1" ||
             Name == "cde.vcx1qa.predicated.v2i64.v4i1" ||
             Name == "cde.vcx2q.predicated.v2i64.v4i1" ||
             Name == "cde.vcx2qa.predicated.v2i64.v4i1" ||
             Name == "cde.vcx3q.predicated.v2i64.v4i1" ||
             Name == "cde.vcx3qa.predicated.v2i64.v4i1") {
    std::vector<Type *> Tys;
    unsigned ID = CI->getIntrinsicID();
    Type *V2I1Ty = FixedVectorType::get(Builder.getInt1Ty(), 2);
    switch (ID) {
    case Intrinsic::arm_mve_mull_int_predicated:
    case Intrinsic::arm_mve_vqdmull_predicated:
    case Intrinsic::arm_mve_vldr_gather_base_predicated:
      Tys = {CI->getType(), CI->getOperand(0)->getType(), V2I1Ty};
      break;
    case Intrinsic::arm_mve_vldr_gather_base_wb_predicated:
    case Intrinsic::arm_mve_vstr_scatter_base_predicated:
    case Intrinsic::arm_mve_vstr_scatter_base_wb_predicated:
      Tys = {CI->getOperand(0)->getType(), CI->getOperand(0)->getType(),
             V2I1Ty};
      break;
    case Intrinsic::arm_mve_vldr_gather_offset_predicated:
      Tys = {CI->getType(), CI->getOperand(0)->getType(),
             CI->getOperand(1)->getType(), V2I1Ty};
      break;
    case Intrinsic::arm_mve_vstr_scatter_offset_predicated:
      Tys = {CI->getOperand(0)->getType(), CI->getOperand(1)->getType(),
             CI->getOperand(2)->getType(), V2I1Ty};
      break;
    case Intrinsic::arm_cde_vcx1q_predicated:
    case Intrinsic::arm_cde_vcx1qa_predicated:
    case Intrinsic::arm_cde_vcx2q_predicated:
    case Intrinsic::arm_cde_vcx2qa_predicated:
    case Intrinsic::arm_cde_vcx3q_predicated:
    case Intrinsic::arm_cde_vcx3qa_predicated:
      Tys = {CI->getOperand(1)->getType(), V2I1Ty};
      break;
    default:
      llvm_unreachable("Unhandled Intrinsic!");
    }

    std::vector<Value *> Ops;
    for (Value *Op : CI->args()) {
      Type *Ty = Op->getType();
      if (Ty->getScalarSizeInBits() == 1) {
        Value *C1 = Builder.CreateCall(
            Intrinsic::getDeclaration(
                F->getParent(), Intrinsic::arm_mve_pred_v2i,
                {VectorType::get(Builder.getInt1Ty(), 4, false)}),
            Op);
        Op = Builder.CreateCall(
            Intrinsic::getDeclaration(F->getParent(),
                                      Intrinsic::arm_mve_pred_i2v, {V2I1Ty}),
            C1);
      }
      Ops.push_back(Op);
    }

    Function *Fn = Intrinsic::getDeclaration(F->getParent(), ID, Tys);
    return Builder.CreateCall(Fn, Ops, CI->getName());
  }
  llvm_unreachable("Unknown function for ARM CallBase upgrade.");
}

static Value *UpgradeAMDGCNIntrinsicCall(StringRef Name, CallBase *CI,
                                         Function *F, IRBuilder<> &Builder) {
  const bool IsInc = Name.startswith("atomic.inc.");
  if (IsInc || Name.startswith("atomic.dec.")) {
    if (CI->getNumOperands() != 6) // Malformed bitcode.
      return nullptr;

    AtomicRMWInst::BinOp RMWOp =
        IsInc ? AtomicRMWInst::UIncWrap : AtomicRMWInst::UDecWrap;

    Value *Ptr = CI->getArgOperand(0);
    Value *Val = CI->getArgOperand(1);
    ConstantInt *OrderArg = dyn_cast<ConstantInt>(CI->getArgOperand(2));
    ConstantInt *VolatileArg = dyn_cast<ConstantInt>(CI->getArgOperand(4));

    AtomicOrdering Order = AtomicOrdering::SequentiallyConsistent;
    if (OrderArg && isValidAtomicOrdering(OrderArg->getZExtValue()))
      Order = static_cast<AtomicOrdering>(OrderArg->getZExtValue());
    if (Order == AtomicOrdering::NotAtomic ||
        Order == AtomicOrdering::Unordered)
      Order = AtomicOrdering::SequentiallyConsistent;

    AtomicRMWInst *RMW = Builder.CreateAtomicRMW(RMWOp, Ptr, Val, std::nullopt, Order);

    if (!VolatileArg || !VolatileArg->isZero())
      RMW->setVolatile(true);
    return RMW;
  }

  llvm_unreachable("Unknown function for AMDGPU intrinsic upgrade.");
}

/// Upgrade a call to an old intrinsic. All argument and return casting must be
/// provided to seamlessly integrate with existing context.
void llvm::UpgradeIntrinsicCall(CallBase *CI, Function *NewFn) {
  // Note dyn_cast to Function is not quite the same as getCalledFunction, which
  // checks the callee's function type matches. It's likely we need to handle
  // type changes here.
  Function *F = dyn_cast<Function>(CI->getCalledOperand());
  if (!F)
    return;

  LLVMContext &C = CI->getContext();
  IRBuilder<> Builder(C);
  Builder.SetInsertPoint(CI->getParent(), CI->getIterator());

  if (!NewFn) {
    // Get the Function's name.
    StringRef Name = F->getName();

    assert(Name.startswith("llvm.") && "Intrinsic doesn't start with 'llvm.'");
    Name = Name.substr(5);

    bool IsX86 = Name.startswith("x86.");
    if (IsX86)
      Name = Name.substr(4);
    bool IsNVVM = Name.startswith("nvvm.");
    if (IsNVVM)
      Name = Name.substr(5);
    bool IsARM = Name.startswith("arm.");
    if (IsARM)
      Name = Name.substr(4);
    bool IsAMDGCN = Name.startswith("amdgcn.");
    if (IsAMDGCN)
      Name = Name.substr(7);

    if (IsX86 && Name.startswith("sse4a.movnt.")) {
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

      StoreInst *SI = Builder.CreateAlignedStore(Extract, Addr, Align(1));
      SI->setMetadata(LLVMContext::MD_nontemporal, Node);

      // Remove intrinsic.
      CI->eraseFromParent();
      return;
    }

    if (IsX86 && (Name.startswith("avx.movnt.") ||
                  Name.startswith("avx512.storent."))) {
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
      StoreInst *SI = Builder.CreateAlignedStore(
          Arg1, BC,
          Align(Arg1->getType()->getPrimitiveSizeInBits().getFixedValue() / 8));
      SI->setMetadata(LLVMContext::MD_nontemporal, Node);

      // Remove intrinsic.
      CI->eraseFromParent();
      return;
    }

    if (IsX86 && Name == "sse2.storel.dq") {
      Value *Arg0 = CI->getArgOperand(0);
      Value *Arg1 = CI->getArgOperand(1);

      auto *NewVecTy = FixedVectorType::get(Type::getInt64Ty(C), 2);
      Value *BC0 = Builder.CreateBitCast(Arg1, NewVecTy, "cast");
      Value *Elt = Builder.CreateExtractElement(BC0, (uint64_t)0);
      Value *BC = Builder.CreateBitCast(Arg0,
                                        PointerType::getUnqual(Elt->getType()),
                                        "cast");
      Builder.CreateAlignedStore(Elt, BC, Align(1));

      // Remove intrinsic.
      CI->eraseFromParent();
      return;
    }

    if (IsX86 && (Name.startswith("sse.storeu.") ||
                  Name.startswith("sse2.storeu.") ||
                  Name.startswith("avx.storeu."))) {
      Value *Arg0 = CI->getArgOperand(0);
      Value *Arg1 = CI->getArgOperand(1);

      Arg0 = Builder.CreateBitCast(Arg0,
                                   PointerType::getUnqual(Arg1->getType()),
                                   "cast");
      Builder.CreateAlignedStore(Arg1, Arg0, Align(1));

      // Remove intrinsic.
      CI->eraseFromParent();
      return;
    }

    if (IsX86 && Name == "avx512.mask.store.ss") {
      Value *Mask = Builder.CreateAnd(CI->getArgOperand(2), Builder.getInt8(1));
      UpgradeMaskedStore(Builder, CI->getArgOperand(0), CI->getArgOperand(1),
                         Mask, false);

      // Remove intrinsic.
      CI->eraseFromParent();
      return;
    }

    if (IsX86 && (Name.startswith("avx512.mask.store"))) {
      // "avx512.mask.storeu." or "avx512.mask.store."
      bool Aligned = Name[17] != 'u'; // "avx512.mask.storeu".
      UpgradeMaskedStore(Builder, CI->getArgOperand(0), CI->getArgOperand(1),
                         CI->getArgOperand(2), Aligned);

      // Remove intrinsic.
      CI->eraseFromParent();
      return;
    }

    Value *Rep;
    // Upgrade packed integer vector compare intrinsics to compare instructions.
    if (IsX86 && (Name.startswith("sse2.pcmp") ||
                  Name.startswith("avx2.pcmp"))) {
      // "sse2.pcpmpeq." "sse2.pcmpgt." "avx2.pcmpeq." or "avx2.pcmpgt."
      bool CmpEq = Name[9] == 'e';
      Rep = Builder.CreateICmp(CmpEq ? ICmpInst::ICMP_EQ : ICmpInst::ICMP_SGT,
                               CI->getArgOperand(0), CI->getArgOperand(1));
      Rep = Builder.CreateSExt(Rep, CI->getType(), "");
    } else if (IsX86 && (Name.startswith("avx512.broadcastm"))) {
      Type *ExtTy = Type::getInt32Ty(C);
      if (CI->getOperand(0)->getType()->isIntegerTy(8))
        ExtTy = Type::getInt64Ty(C);
      unsigned NumElts = CI->getType()->getPrimitiveSizeInBits() /
                         ExtTy->getPrimitiveSizeInBits();
      Rep = Builder.CreateZExt(CI->getArgOperand(0), ExtTy);
      Rep = Builder.CreateVectorSplat(NumElts, Rep);
    } else if (IsX86 && (Name == "sse.sqrt.ss" ||
                         Name == "sse2.sqrt.sd")) {
      Value *Vec = CI->getArgOperand(0);
      Value *Elt0 = Builder.CreateExtractElement(Vec, (uint64_t)0);
      Function *Intr = Intrinsic::getDeclaration(F->getParent(),
                                                 Intrinsic::sqrt, Elt0->getType());
      Elt0 = Builder.CreateCall(Intr, Elt0);
      Rep = Builder.CreateInsertElement(Vec, Elt0, (uint64_t)0);
    } else if (IsX86 && (Name.startswith("avx.sqrt.p") ||
                         Name.startswith("sse2.sqrt.p") ||
                         Name.startswith("sse.sqrt.p"))) {
      Rep = Builder.CreateCall(Intrinsic::getDeclaration(F->getParent(),
                                                         Intrinsic::sqrt,
                                                         CI->getType()),
                               {CI->getArgOperand(0)});
    } else if (IsX86 && (Name.startswith("avx512.mask.sqrt.p"))) {
      if (CI->arg_size() == 4 &&
          (!isa<ConstantInt>(CI->getArgOperand(3)) ||
           cast<ConstantInt>(CI->getArgOperand(3))->getZExtValue() != 4)) {
        Intrinsic::ID IID = Name[18] == 's' ? Intrinsic::x86_avx512_sqrt_ps_512
                                            : Intrinsic::x86_avx512_sqrt_pd_512;

        Value *Args[] = { CI->getArgOperand(0), CI->getArgOperand(3) };
        Rep = Builder.CreateCall(Intrinsic::getDeclaration(CI->getModule(),
                                                           IID), Args);
      } else {
        Rep = Builder.CreateCall(Intrinsic::getDeclaration(F->getParent(),
                                                           Intrinsic::sqrt,
                                                           CI->getType()),
                                 {CI->getArgOperand(0)});
      }
      Rep = EmitX86Select(Builder, CI->getArgOperand(2), Rep,
                          CI->getArgOperand(1));
    } else if (IsX86 && (Name.startswith("avx512.ptestm") ||
                         Name.startswith("avx512.ptestnm"))) {
      Value *Op0 = CI->getArgOperand(0);
      Value *Op1 = CI->getArgOperand(1);
      Value *Mask = CI->getArgOperand(2);
      Rep = Builder.CreateAnd(Op0, Op1);
      llvm::Type *Ty = Op0->getType();
      Value *Zero = llvm::Constant::getNullValue(Ty);
      ICmpInst::Predicate Pred =
        Name.startswith("avx512.ptestm") ? ICmpInst::ICMP_NE : ICmpInst::ICMP_EQ;
      Rep = Builder.CreateICmp(Pred, Rep, Zero);
      Rep = ApplyX86MaskOn1BitsVec(Builder, Rep, Mask);
    } else if (IsX86 && (Name.startswith("avx512.mask.pbroadcast"))){
      unsigned NumElts = cast<FixedVectorType>(CI->getArgOperand(1)->getType())
                             ->getNumElements();
      Rep = Builder.CreateVectorSplat(NumElts, CI->getArgOperand(0));
      Rep = EmitX86Select(Builder, CI->getArgOperand(2), Rep,
                          CI->getArgOperand(1));
    } else if (IsX86 && (Name.startswith("avx512.kunpck"))) {
      unsigned NumElts = CI->getType()->getScalarSizeInBits();
      Value *LHS = getX86MaskVec(Builder, CI->getArgOperand(0), NumElts);
      Value *RHS = getX86MaskVec(Builder, CI->getArgOperand(1), NumElts);
      int Indices[64];
      for (unsigned i = 0; i != NumElts; ++i)
        Indices[i] = i;

      // First extract half of each vector. This gives better codegen than
      // doing it in a single shuffle.
      LHS =
          Builder.CreateShuffleVector(LHS, LHS, ArrayRef(Indices, NumElts / 2));
      RHS =
          Builder.CreateShuffleVector(RHS, RHS, ArrayRef(Indices, NumElts / 2));
      // Concat the vectors.
      // NOTE: Operands have to be swapped to match intrinsic definition.
      Rep = Builder.CreateShuffleVector(RHS, LHS, ArrayRef(Indices, NumElts));
      Rep = Builder.CreateBitCast(Rep, CI->getType());
    } else if (IsX86 && Name == "avx512.kand.w") {
      Value *LHS = getX86MaskVec(Builder, CI->getArgOperand(0), 16);
      Value *RHS = getX86MaskVec(Builder, CI->getArgOperand(1), 16);
      Rep = Builder.CreateAnd(LHS, RHS);
      Rep = Builder.CreateBitCast(Rep, CI->getType());
    } else if (IsX86 && Name == "avx512.kandn.w") {
      Value *LHS = getX86MaskVec(Builder, CI->getArgOperand(0), 16);
      Value *RHS = getX86MaskVec(Builder, CI->getArgOperand(1), 16);
      LHS = Builder.CreateNot(LHS);
      Rep = Builder.CreateAnd(LHS, RHS);
      Rep = Builder.CreateBitCast(Rep, CI->getType());
    } else if (IsX86 && Name == "avx512.kor.w") {
      Value *LHS = getX86MaskVec(Builder, CI->getArgOperand(0), 16);
      Value *RHS = getX86MaskVec(Builder, CI->getArgOperand(1), 16);
      Rep = Builder.CreateOr(LHS, RHS);
      Rep = Builder.CreateBitCast(Rep, CI->getType());
    } else if (IsX86 && Name == "avx512.kxor.w") {
      Value *LHS = getX86MaskVec(Builder, CI->getArgOperand(0), 16);
      Value *RHS = getX86MaskVec(Builder, CI->getArgOperand(1), 16);
      Rep = Builder.CreateXor(LHS, RHS);
      Rep = Builder.CreateBitCast(Rep, CI->getType());
    } else if (IsX86 && Name == "avx512.kxnor.w") {
      Value *LHS = getX86MaskVec(Builder, CI->getArgOperand(0), 16);
      Value *RHS = getX86MaskVec(Builder, CI->getArgOperand(1), 16);
      LHS = Builder.CreateNot(LHS);
      Rep = Builder.CreateXor(LHS, RHS);
      Rep = Builder.CreateBitCast(Rep, CI->getType());
    } else if (IsX86 && Name == "avx512.knot.w") {
      Rep = getX86MaskVec(Builder, CI->getArgOperand(0), 16);
      Rep = Builder.CreateNot(Rep);
      Rep = Builder.CreateBitCast(Rep, CI->getType());
    } else if (IsX86 &&
               (Name == "avx512.kortestz.w" || Name == "avx512.kortestc.w")) {
      Value *LHS = getX86MaskVec(Builder, CI->getArgOperand(0), 16);
      Value *RHS = getX86MaskVec(Builder, CI->getArgOperand(1), 16);
      Rep = Builder.CreateOr(LHS, RHS);
      Rep = Builder.CreateBitCast(Rep, Builder.getInt16Ty());
      Value *C;
      if (Name[14] == 'c')
        C = ConstantInt::getAllOnesValue(Builder.getInt16Ty());
      else
        C = ConstantInt::getNullValue(Builder.getInt16Ty());
      Rep = Builder.CreateICmpEQ(Rep, C);
      Rep = Builder.CreateZExt(Rep, Builder.getInt32Ty());
    } else if (IsX86 && (Name == "sse.add.ss" || Name == "sse2.add.sd" ||
                         Name == "sse.sub.ss" || Name == "sse2.sub.sd" ||
                         Name == "sse.mul.ss" || Name == "sse2.mul.sd" ||
                         Name == "sse.div.ss" || Name == "sse2.div.sd")) {
      Type *I32Ty = Type::getInt32Ty(C);
      Value *Elt0 = Builder.CreateExtractElement(CI->getArgOperand(0),
                                                 ConstantInt::get(I32Ty, 0));
      Value *Elt1 = Builder.CreateExtractElement(CI->getArgOperand(1),
                                                 ConstantInt::get(I32Ty, 0));
      Value *EltOp;
      if (Name.contains(".add."))
        EltOp = Builder.CreateFAdd(Elt0, Elt1);
      else if (Name.contains(".sub."))
        EltOp = Builder.CreateFSub(Elt0, Elt1);
      else if (Name.contains(".mul."))
        EltOp = Builder.CreateFMul(Elt0, Elt1);
      else
        EltOp = Builder.CreateFDiv(Elt0, Elt1);
      Rep = Builder.CreateInsertElement(CI->getArgOperand(0), EltOp,
                                        ConstantInt::get(I32Ty, 0));
    } else if (IsX86 && Name.startswith("avx512.mask.pcmp")) {
      // "avx512.mask.pcmpeq." or "avx512.mask.pcmpgt."
      bool CmpEq = Name[16] == 'e';
      Rep = upgradeMaskedCompare(Builder, *CI, CmpEq ? 0 : 6, true);
    } else if (IsX86 && Name.startswith("avx512.mask.vpshufbitqmb.")) {
      Type *OpTy = CI->getArgOperand(0)->getType();
      unsigned VecWidth = OpTy->getPrimitiveSizeInBits();
      Intrinsic::ID IID;
      switch (VecWidth) {
      default: llvm_unreachable("Unexpected intrinsic");
      case 128: IID = Intrinsic::x86_avx512_vpshufbitqmb_128; break;
      case 256: IID = Intrinsic::x86_avx512_vpshufbitqmb_256; break;
      case 512: IID = Intrinsic::x86_avx512_vpshufbitqmb_512; break;
      }

      Rep = Builder.CreateCall(Intrinsic::getDeclaration(F->getParent(), IID),
                               { CI->getOperand(0), CI->getArgOperand(1) });
      Rep = ApplyX86MaskOn1BitsVec(Builder, Rep, CI->getArgOperand(2));
    } else if (IsX86 && Name.startswith("avx512.mask.fpclass.p")) {
      Type *OpTy = CI->getArgOperand(0)->getType();
      unsigned VecWidth = OpTy->getPrimitiveSizeInBits();
      unsigned EltWidth = OpTy->getScalarSizeInBits();
      Intrinsic::ID IID;
      if (VecWidth == 128 && EltWidth == 32)
        IID = Intrinsic::x86_avx512_fpclass_ps_128;
      else if (VecWidth == 256 && EltWidth == 32)
        IID = Intrinsic::x86_avx512_fpclass_ps_256;
      else if (VecWidth == 512 && EltWidth == 32)
        IID = Intrinsic::x86_avx512_fpclass_ps_512;
      else if (VecWidth == 128 && EltWidth == 64)
        IID = Intrinsic::x86_avx512_fpclass_pd_128;
      else if (VecWidth == 256 && EltWidth == 64)
        IID = Intrinsic::x86_avx512_fpclass_pd_256;
      else if (VecWidth == 512 && EltWidth == 64)
        IID = Intrinsic::x86_avx512_fpclass_pd_512;
      else
        llvm_unreachable("Unexpected intrinsic");

      Rep = Builder.CreateCall(Intrinsic::getDeclaration(F->getParent(), IID),
                               { CI->getOperand(0), CI->getArgOperand(1) });
      Rep = ApplyX86MaskOn1BitsVec(Builder, Rep, CI->getArgOperand(2));
    } else if (IsX86 && Name.startswith("avx512.cmp.p")) {
      SmallVector<Value *, 4> Args(CI->args());
      Type *OpTy = Args[0]->getType();
      unsigned VecWidth = OpTy->getPrimitiveSizeInBits();
      unsigned EltWidth = OpTy->getScalarSizeInBits();
      Intrinsic::ID IID;
      if (VecWidth == 128 && EltWidth == 32)
        IID = Intrinsic::x86_avx512_mask_cmp_ps_128;
      else if (VecWidth == 256 && EltWidth == 32)
        IID = Intrinsic::x86_avx512_mask_cmp_ps_256;
      else if (VecWidth == 512 && EltWidth == 32)
        IID = Intrinsic::x86_avx512_mask_cmp_ps_512;
      else if (VecWidth == 128 && EltWidth == 64)
        IID = Intrinsic::x86_avx512_mask_cmp_pd_128;
      else if (VecWidth == 256 && EltWidth == 64)
        IID = Intrinsic::x86_avx512_mask_cmp_pd_256;
      else if (VecWidth == 512 && EltWidth == 64)
        IID = Intrinsic::x86_avx512_mask_cmp_pd_512;
      else
        llvm_unreachable("Unexpected intrinsic");

      Value *Mask = Constant::getAllOnesValue(CI->getType());
      if (VecWidth == 512)
        std::swap(Mask, Args.back());
      Args.push_back(Mask);

      Rep = Builder.CreateCall(Intrinsic::getDeclaration(F->getParent(), IID),
                               Args);
    } else if (IsX86 && Name.startswith("avx512.mask.cmp.")) {
      // Integer compare intrinsics.
      unsigned Imm = cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue();
      Rep = upgradeMaskedCompare(Builder, *CI, Imm, true);
    } else if (IsX86 && Name.startswith("avx512.mask.ucmp.")) {
      unsigned Imm = cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue();
      Rep = upgradeMaskedCompare(Builder, *CI, Imm, false);
    } else if (IsX86 && (Name.startswith("avx512.cvtb2mask.") ||
                         Name.startswith("avx512.cvtw2mask.") ||
                         Name.startswith("avx512.cvtd2mask.") ||
                         Name.startswith("avx512.cvtq2mask."))) {
      Value *Op = CI->getArgOperand(0);
      Value *Zero = llvm::Constant::getNullValue(Op->getType());
      Rep = Builder.CreateICmp(ICmpInst::ICMP_SLT, Op, Zero);
      Rep = ApplyX86MaskOn1BitsVec(Builder, Rep, nullptr);
    } else if(IsX86 && (Name == "ssse3.pabs.b.128" ||
                        Name == "ssse3.pabs.w.128" ||
                        Name == "ssse3.pabs.d.128" ||
                        Name.startswith("avx2.pabs") ||
                        Name.startswith("avx512.mask.pabs"))) {
      Rep = upgradeAbs(Builder, *CI);
    } else if (IsX86 && (Name == "sse41.pmaxsb" ||
                         Name == "sse2.pmaxs.w" ||
                         Name == "sse41.pmaxsd" ||
                         Name.startswith("avx2.pmaxs") ||
                         Name.startswith("avx512.mask.pmaxs"))) {
      Rep = UpgradeX86BinaryIntrinsics(Builder, *CI, Intrinsic::smax);
    } else if (IsX86 && (Name == "sse2.pmaxu.b" ||
                         Name == "sse41.pmaxuw" ||
                         Name == "sse41.pmaxud" ||
                         Name.startswith("avx2.pmaxu") ||
                         Name.startswith("avx512.mask.pmaxu"))) {
      Rep = UpgradeX86BinaryIntrinsics(Builder, *CI, Intrinsic::umax);
    } else if (IsX86 && (Name == "sse41.pminsb" ||
                         Name == "sse2.pmins.w" ||
                         Name == "sse41.pminsd" ||
                         Name.startswith("avx2.pmins") ||
                         Name.startswith("avx512.mask.pmins"))) {
      Rep = UpgradeX86BinaryIntrinsics(Builder, *CI, Intrinsic::smin);
    } else if (IsX86 && (Name == "sse2.pminu.b" ||
                         Name == "sse41.pminuw" ||
                         Name == "sse41.pminud" ||
                         Name.startswith("avx2.pminu") ||
                         Name.startswith("avx512.mask.pminu"))) {
      Rep = UpgradeX86BinaryIntrinsics(Builder, *CI, Intrinsic::umin);
    } else if (IsX86 && (Name == "sse2.pmulu.dq" ||
                         Name == "avx2.pmulu.dq" ||
                         Name == "avx512.pmulu.dq.512" ||
                         Name.startswith("avx512.mask.pmulu.dq."))) {
      Rep = upgradePMULDQ(Builder, *CI, /*Signed*/false);
    } else if (IsX86 && (Name == "sse41.pmuldq" ||
                         Name == "avx2.pmul.dq" ||
                         Name == "avx512.pmul.dq.512" ||
                         Name.startswith("avx512.mask.pmul.dq."))) {
      Rep = upgradePMULDQ(Builder, *CI, /*Signed*/true);
    } else if (IsX86 && (Name == "sse.cvtsi2ss" ||
                         Name == "sse2.cvtsi2sd" ||
                         Name == "sse.cvtsi642ss" ||
                         Name == "sse2.cvtsi642sd")) {
      Rep = Builder.CreateSIToFP(
          CI->getArgOperand(1),
          cast<VectorType>(CI->getType())->getElementType());
      Rep = Builder.CreateInsertElement(CI->getArgOperand(0), Rep, (uint64_t)0);
    } else if (IsX86 && Name == "avx512.cvtusi2sd") {
      Rep = Builder.CreateUIToFP(
          CI->getArgOperand(1),
          cast<VectorType>(CI->getType())->getElementType());
      Rep = Builder.CreateInsertElement(CI->getArgOperand(0), Rep, (uint64_t)0);
    } else if (IsX86 && Name == "sse2.cvtss2sd") {
      Rep = Builder.CreateExtractElement(CI->getArgOperand(1), (uint64_t)0);
      Rep = Builder.CreateFPExt(
          Rep, cast<VectorType>(CI->getType())->getElementType());
      Rep = Builder.CreateInsertElement(CI->getArgOperand(0), Rep, (uint64_t)0);
    } else if (IsX86 && (Name == "sse2.cvtdq2pd" ||
                         Name == "sse2.cvtdq2ps" ||
                         Name == "avx.cvtdq2.pd.256" ||
                         Name == "avx.cvtdq2.ps.256" ||
                         Name.startswith("avx512.mask.cvtdq2pd.") ||
                         Name.startswith("avx512.mask.cvtudq2pd.") ||
                         Name.startswith("avx512.mask.cvtdq2ps.") ||
                         Name.startswith("avx512.mask.cvtudq2ps.") ||
                         Name.startswith("avx512.mask.cvtqq2pd.") ||
                         Name.startswith("avx512.mask.cvtuqq2pd.") ||
                         Name == "avx512.mask.cvtqq2ps.256" ||
                         Name == "avx512.mask.cvtqq2ps.512" ||
                         Name == "avx512.mask.cvtuqq2ps.256" ||
                         Name == "avx512.mask.cvtuqq2ps.512" ||
                         Name == "sse2.cvtps2pd" ||
                         Name == "avx.cvt.ps2.pd.256" ||
                         Name == "avx512.mask.cvtps2pd.128" ||
                         Name == "avx512.mask.cvtps2pd.256")) {
      auto *DstTy = cast<FixedVectorType>(CI->getType());
      Rep = CI->getArgOperand(0);
      auto *SrcTy = cast<FixedVectorType>(Rep->getType());

      unsigned NumDstElts = DstTy->getNumElements();
      if (NumDstElts < SrcTy->getNumElements()) {
        assert(NumDstElts == 2 && "Unexpected vector size");
        Rep = Builder.CreateShuffleVector(Rep, Rep, ArrayRef<int>{0, 1});
      }

      bool IsPS2PD = SrcTy->getElementType()->isFloatTy();
      bool IsUnsigned = (StringRef::npos != Name.find("cvtu"));
      if (IsPS2PD)
        Rep = Builder.CreateFPExt(Rep, DstTy, "cvtps2pd");
      else if (CI->arg_size() == 4 &&
               (!isa<ConstantInt>(CI->getArgOperand(3)) ||
                cast<ConstantInt>(CI->getArgOperand(3))->getZExtValue() != 4)) {
        Intrinsic::ID IID = IsUnsigned ? Intrinsic::x86_avx512_uitofp_round
                                       : Intrinsic::x86_avx512_sitofp_round;
        Function *F = Intrinsic::getDeclaration(CI->getModule(), IID,
                                                { DstTy, SrcTy });
        Rep = Builder.CreateCall(F, { Rep, CI->getArgOperand(3) });
      } else {
        Rep = IsUnsigned ? Builder.CreateUIToFP(Rep, DstTy, "cvt")
                         : Builder.CreateSIToFP(Rep, DstTy, "cvt");
      }

      if (CI->arg_size() >= 3)
        Rep = EmitX86Select(Builder, CI->getArgOperand(2), Rep,
                            CI->getArgOperand(1));
    } else if (IsX86 && (Name.startswith("avx512.mask.vcvtph2ps.") ||
                         Name.startswith("vcvtph2ps."))) {
      auto *DstTy = cast<FixedVectorType>(CI->getType());
      Rep = CI->getArgOperand(0);
      auto *SrcTy = cast<FixedVectorType>(Rep->getType());
      unsigned NumDstElts = DstTy->getNumElements();
      if (NumDstElts != SrcTy->getNumElements()) {
        assert(NumDstElts == 4 && "Unexpected vector size");
        Rep = Builder.CreateShuffleVector(Rep, Rep, ArrayRef<int>{0, 1, 2, 3});
      }
      Rep = Builder.CreateBitCast(
          Rep, FixedVectorType::get(Type::getHalfTy(C), NumDstElts));
      Rep = Builder.CreateFPExt(Rep, DstTy, "cvtph2ps");
      if (CI->arg_size() >= 3)
        Rep = EmitX86Select(Builder, CI->getArgOperand(2), Rep,
                            CI->getArgOperand(1));
    } else if (IsX86 && Name.startswith("avx512.mask.load")) {
      // "avx512.mask.loadu." or "avx512.mask.load."
      bool Aligned = Name[16] != 'u'; // "avx512.mask.loadu".
      Rep =
          UpgradeMaskedLoad(Builder, CI->getArgOperand(0), CI->getArgOperand(1),
                            CI->getArgOperand(2), Aligned);
    } else if (IsX86 && Name.startswith("avx512.mask.expand.load.")) {
      auto *ResultTy = cast<FixedVectorType>(CI->getType());
      Type *PtrTy = ResultTy->getElementType();

      // Cast the pointer to element type.
      Value *Ptr = Builder.CreateBitCast(CI->getOperand(0),
                                         llvm::PointerType::getUnqual(PtrTy));

      Value *MaskVec = getX86MaskVec(Builder, CI->getArgOperand(2),
                                     ResultTy->getNumElements());

      Function *ELd = Intrinsic::getDeclaration(F->getParent(),
                                                Intrinsic::masked_expandload,
                                                ResultTy);
      Rep = Builder.CreateCall(ELd, { Ptr, MaskVec, CI->getOperand(1) });
    } else if (IsX86 && Name.startswith("avx512.mask.compress.store.")) {
      auto *ResultTy = cast<VectorType>(CI->getArgOperand(1)->getType());
      Type *PtrTy = ResultTy->getElementType();

      // Cast the pointer to element type.
      Value *Ptr = Builder.CreateBitCast(CI->getOperand(0),
                                         llvm::PointerType::getUnqual(PtrTy));

      Value *MaskVec =
          getX86MaskVec(Builder, CI->getArgOperand(2),
                        cast<FixedVectorType>(ResultTy)->getNumElements());

      Function *CSt = Intrinsic::getDeclaration(F->getParent(),
                                                Intrinsic::masked_compressstore,
                                                ResultTy);
      Rep = Builder.CreateCall(CSt, { CI->getArgOperand(1), Ptr, MaskVec });
    } else if (IsX86 && (Name.startswith("avx512.mask.compress.") ||
                         Name.startswith("avx512.mask.expand."))) {
      auto *ResultTy = cast<FixedVectorType>(CI->getType());

      Value *MaskVec = getX86MaskVec(Builder, CI->getArgOperand(2),
                                     ResultTy->getNumElements());

      bool IsCompress = Name[12] == 'c';
      Intrinsic::ID IID = IsCompress ? Intrinsic::x86_avx512_mask_compress
                                     : Intrinsic::x86_avx512_mask_expand;
      Function *Intr = Intrinsic::getDeclaration(F->getParent(), IID, ResultTy);
      Rep = Builder.CreateCall(Intr, { CI->getOperand(0), CI->getOperand(1),
                                       MaskVec });
    } else if (IsX86 && Name.startswith("xop.vpcom")) {
      bool IsSigned;
      if (Name.endswith("ub") || Name.endswith("uw") || Name.endswith("ud") ||
          Name.endswith("uq"))
        IsSigned = false;
      else if (Name.endswith("b") || Name.endswith("w") || Name.endswith("d") ||
               Name.endswith("q"))
        IsSigned = true;
      else
        llvm_unreachable("Unknown suffix");

      unsigned Imm;
      if (CI->arg_size() == 3) {
        Imm = cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue();
      } else {
        Name = Name.substr(9); // strip off "xop.vpcom"
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
      }

      Rep = upgradeX86vpcom(Builder, *CI, Imm, IsSigned);
    } else if (IsX86 && Name.startswith("xop.vpcmov")) {
      Value *Sel = CI->getArgOperand(2);
      Value *NotSel = Builder.CreateNot(Sel);
      Value *Sel0 = Builder.CreateAnd(CI->getArgOperand(0), Sel);
      Value *Sel1 = Builder.CreateAnd(CI->getArgOperand(1), NotSel);
      Rep = Builder.CreateOr(Sel0, Sel1);
    } else if (IsX86 && (Name.startswith("xop.vprot") ||
                         Name.startswith("avx512.prol") ||
                         Name.startswith("avx512.mask.prol"))) {
      Rep = upgradeX86Rotate(Builder, *CI, false);
    } else if (IsX86 && (Name.startswith("avx512.pror") ||
                         Name.startswith("avx512.mask.pror"))) {
      Rep = upgradeX86Rotate(Builder, *CI, true);
    } else if (IsX86 && (Name.startswith("avx512.vpshld.") ||
                         Name.startswith("avx512.mask.vpshld") ||
                         Name.startswith("avx512.maskz.vpshld"))) {
      bool ZeroMask = Name[11] == 'z';
      Rep = upgradeX86ConcatShift(Builder, *CI, false, ZeroMask);
    } else if (IsX86 && (Name.startswith("avx512.vpshrd.") ||
                         Name.startswith("avx512.mask.vpshrd") ||
                         Name.startswith("avx512.maskz.vpshrd"))) {
      bool ZeroMask = Name[11] == 'z';
      Rep = upgradeX86ConcatShift(Builder, *CI, true, ZeroMask);
    } else if (IsX86 && Name == "sse42.crc32.64.8") {
      Function *CRC32 = Intrinsic::getDeclaration(F->getParent(),
                                               Intrinsic::x86_sse42_crc32_32_8);
      Value *Trunc0 = Builder.CreateTrunc(CI->getArgOperand(0), Type::getInt32Ty(C));
      Rep = Builder.CreateCall(CRC32, {Trunc0, CI->getArgOperand(1)});
      Rep = Builder.CreateZExt(Rep, CI->getType(), "");
    } else if (IsX86 && (Name.startswith("avx.vbroadcast.s") ||
                         Name.startswith("avx512.vbroadcast.s"))) {
      // Replace broadcasts with a series of insertelements.
      auto *VecTy = cast<FixedVectorType>(CI->getType());
      Type *EltTy = VecTy->getElementType();
      unsigned EltNum = VecTy->getNumElements();
      Value *Cast = Builder.CreateBitCast(CI->getArgOperand(0),
                                          EltTy->getPointerTo());
      Value *Load = Builder.CreateLoad(EltTy, Cast);
      Type *I32Ty = Type::getInt32Ty(C);
      Rep = PoisonValue::get(VecTy);
      for (unsigned I = 0; I < EltNum; ++I)
        Rep = Builder.CreateInsertElement(Rep, Load,
                                          ConstantInt::get(I32Ty, I));
    } else if (IsX86 && (Name.startswith("sse41.pmovsx") ||
                         Name.startswith("sse41.pmovzx") ||
                         Name.startswith("avx2.pmovsx") ||
                         Name.startswith("avx2.pmovzx") ||
                         Name.startswith("avx512.mask.pmovsx") ||
                         Name.startswith("avx512.mask.pmovzx"))) {
      auto *DstTy = cast<FixedVectorType>(CI->getType());
      unsigned NumDstElts = DstTy->getNumElements();

      // Extract a subvector of the first NumDstElts lanes and sign/zero extend.
      SmallVector<int, 8> ShuffleMask(NumDstElts);
      for (unsigned i = 0; i != NumDstElts; ++i)
        ShuffleMask[i] = i;

      Value *SV =
          Builder.CreateShuffleVector(CI->getArgOperand(0), ShuffleMask);

      bool DoSext = (StringRef::npos != Name.find("pmovsx"));
      Rep = DoSext ? Builder.CreateSExt(SV, DstTy)
                   : Builder.CreateZExt(SV, DstTy);
      // If there are 3 arguments, it's a masked intrinsic so we need a select.
      if (CI->arg_size() == 3)
        Rep = EmitX86Select(Builder, CI->getArgOperand(2), Rep,
                            CI->getArgOperand(1));
    } else if (Name == "avx512.mask.pmov.qd.256" ||
               Name == "avx512.mask.pmov.qd.512" ||
               Name == "avx512.mask.pmov.wb.256" ||
               Name == "avx512.mask.pmov.wb.512") {
      Type *Ty = CI->getArgOperand(1)->getType();
      Rep = Builder.CreateTrunc(CI->getArgOperand(0), Ty);
      Rep = EmitX86Select(Builder, CI->getArgOperand(2), Rep,
                          CI->getArgOperand(1));
    } else if (IsX86 && (Name.startswith("avx.vbroadcastf128") ||
                         Name == "avx2.vbroadcasti128")) {
      // Replace vbroadcastf128/vbroadcasti128 with a vector load+shuffle.
      Type *EltTy = cast<VectorType>(CI->getType())->getElementType();
      unsigned NumSrcElts = 128 / EltTy->getPrimitiveSizeInBits();
      auto *VT = FixedVectorType::get(EltTy, NumSrcElts);
      Value *Op = Builder.CreatePointerCast(CI->getArgOperand(0),
                                            PointerType::getUnqual(VT));
      Value *Load = Builder.CreateAlignedLoad(VT, Op, Align(1));
      if (NumSrcElts == 2)
        Rep = Builder.CreateShuffleVector(Load, ArrayRef<int>{0, 1, 0, 1});
      else
        Rep = Builder.CreateShuffleVector(
            Load, ArrayRef<int>{0, 1, 2, 3, 0, 1, 2, 3});
    } else if (IsX86 && (Name.startswith("avx512.mask.shuf.i") ||
                         Name.startswith("avx512.mask.shuf.f"))) {
      unsigned Imm = cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue();
      Type *VT = CI->getType();
      unsigned NumLanes = VT->getPrimitiveSizeInBits() / 128;
      unsigned NumElementsInLane = 128 / VT->getScalarSizeInBits();
      unsigned ControlBitsMask = NumLanes - 1;
      unsigned NumControlBits = NumLanes / 2;
      SmallVector<int, 8> ShuffleMask(0);

      for (unsigned l = 0; l != NumLanes; ++l) {
        unsigned LaneMask = (Imm >> (l * NumControlBits)) & ControlBitsMask;
        // We actually need the other source.
        if (l >= NumLanes / 2)
          LaneMask += NumLanes;
        for (unsigned i = 0; i != NumElementsInLane; ++i)
          ShuffleMask.push_back(LaneMask * NumElementsInLane + i);
      }
      Rep = Builder.CreateShuffleVector(CI->getArgOperand(0),
                                        CI->getArgOperand(1), ShuffleMask);
      Rep = EmitX86Select(Builder, CI->getArgOperand(4), Rep,
                          CI->getArgOperand(3));
    }else if (IsX86 && (Name.startswith("avx512.mask.broadcastf") ||
                         Name.startswith("avx512.mask.broadcasti"))) {
      unsigned NumSrcElts =
          cast<FixedVectorType>(CI->getArgOperand(0)->getType())
              ->getNumElements();
      unsigned NumDstElts =
          cast<FixedVectorType>(CI->getType())->getNumElements();

      SmallVector<int, 8> ShuffleMask(NumDstElts);
      for (unsigned i = 0; i != NumDstElts; ++i)
        ShuffleMask[i] = i % NumSrcElts;

      Rep = Builder.CreateShuffleVector(CI->getArgOperand(0),
                                        CI->getArgOperand(0),
                                        ShuffleMask);
      Rep = EmitX86Select(Builder, CI->getArgOperand(2), Rep,
                          CI->getArgOperand(1));
    } else if (IsX86 && (Name.startswith("avx2.pbroadcast") ||
                         Name.startswith("avx2.vbroadcast") ||
                         Name.startswith("avx512.pbroadcast") ||
                         Name.startswith("avx512.mask.broadcast.s"))) {
      // Replace vp?broadcasts with a vector shuffle.
      Value *Op = CI->getArgOperand(0);
      ElementCount EC = cast<VectorType>(CI->getType())->getElementCount();
      Type *MaskTy = VectorType::get(Type::getInt32Ty(C), EC);
      SmallVector<int, 8> M;
      ShuffleVectorInst::getShuffleMask(Constant::getNullValue(MaskTy), M);
      Rep = Builder.CreateShuffleVector(Op, M);

      if (CI->arg_size() == 3)
        Rep = EmitX86Select(Builder, CI->getArgOperand(2), Rep,
                            CI->getArgOperand(1));
    } else if (IsX86 && (Name.startswith("sse2.padds.") ||
                         Name.startswith("avx2.padds.") ||
                         Name.startswith("avx512.padds.") ||
                         Name.startswith("avx512.mask.padds."))) {
      Rep = UpgradeX86BinaryIntrinsics(Builder, *CI, Intrinsic::sadd_sat);
    } else if (IsX86 && (Name.startswith("sse2.psubs.") ||
                         Name.startswith("avx2.psubs.") ||
                         Name.startswith("avx512.psubs.") ||
                         Name.startswith("avx512.mask.psubs."))) {
      Rep = UpgradeX86BinaryIntrinsics(Builder, *CI, Intrinsic::ssub_sat);
    } else if (IsX86 && (Name.startswith("sse2.paddus.") ||
                         Name.startswith("avx2.paddus.") ||
                         Name.startswith("avx512.mask.paddus."))) {
      Rep = UpgradeX86BinaryIntrinsics(Builder, *CI, Intrinsic::uadd_sat);
    } else if (IsX86 && (Name.startswith("sse2.psubus.") ||
                         Name.startswith("avx2.psubus.") ||
                         Name.startswith("avx512.mask.psubus."))) {
      Rep = UpgradeX86BinaryIntrinsics(Builder, *CI, Intrinsic::usub_sat);
    } else if (IsX86 && Name.startswith("avx512.mask.palignr.")) {
      Rep = UpgradeX86ALIGNIntrinsics(Builder, CI->getArgOperand(0),
                                      CI->getArgOperand(1),
                                      CI->getArgOperand(2),
                                      CI->getArgOperand(3),
                                      CI->getArgOperand(4),
                                      false);
    } else if (IsX86 && Name.startswith("avx512.mask.valign.")) {
      Rep = UpgradeX86ALIGNIntrinsics(Builder, CI->getArgOperand(0),
                                      CI->getArgOperand(1),
                                      CI->getArgOperand(2),
                                      CI->getArgOperand(3),
                                      CI->getArgOperand(4),
                                      true);
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
      auto *VecTy = cast<FixedVectorType>(CI->getType());
      unsigned NumElts = VecTy->getNumElements();

      SmallVector<int, 16> Idxs(NumElts);
      for (unsigned i = 0; i != NumElts; ++i)
        Idxs[i] = ((Imm >> (i%8)) & 1) ? i + NumElts : i;

      Rep = Builder.CreateShuffleVector(Op0, Op1, Idxs);
    } else if (IsX86 && (Name.startswith("avx.vinsertf128.") ||
                         Name == "avx2.vinserti128" ||
                         Name.startswith("avx512.mask.insert"))) {
      Value *Op0 = CI->getArgOperand(0);
      Value *Op1 = CI->getArgOperand(1);
      unsigned Imm = cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue();
      unsigned DstNumElts =
          cast<FixedVectorType>(CI->getType())->getNumElements();
      unsigned SrcNumElts =
          cast<FixedVectorType>(Op1->getType())->getNumElements();
      unsigned Scale = DstNumElts / SrcNumElts;

      // Mask off the high bits of the immediate value; hardware ignores those.
      Imm = Imm % Scale;

      // Extend the second operand into a vector the size of the destination.
      SmallVector<int, 8> Idxs(DstNumElts);
      for (unsigned i = 0; i != SrcNumElts; ++i)
        Idxs[i] = i;
      for (unsigned i = SrcNumElts; i != DstNumElts; ++i)
        Idxs[i] = SrcNumElts;
      Rep = Builder.CreateShuffleVector(Op1, Idxs);

      // Insert the second operand into the first operand.

      // Note that there is no guarantee that instruction lowering will actually
      // produce a vinsertf128 instruction for the created shuffles. In
      // particular, the 0 immediate case involves no lane changes, so it can
      // be handled as a blend.

      // Example of shuffle mask for 32-bit elements:
      // Imm = 1  <i32 0, i32 1, i32 2,  i32 3,  i32 8, i32 9, i32 10, i32 11>
      // Imm = 0  <i32 8, i32 9, i32 10, i32 11, i32 4, i32 5, i32 6,  i32 7 >

      // First fill with identify mask.
      for (unsigned i = 0; i != DstNumElts; ++i)
        Idxs[i] = i;
      // Then replace the elements where we need to insert.
      for (unsigned i = 0; i != SrcNumElts; ++i)
        Idxs[i + Imm * SrcNumElts] = i + DstNumElts;
      Rep = Builder.CreateShuffleVector(Op0, Rep, Idxs);

      // If the intrinsic has a mask operand, handle that.
      if (CI->arg_size() == 5)
        Rep = EmitX86Select(Builder, CI->getArgOperand(4), Rep,
                            CI->getArgOperand(3));
    } else if (IsX86 && (Name.startswith("avx.vextractf128.") ||
                         Name == "avx2.vextracti128" ||
                         Name.startswith("avx512.mask.vextract"))) {
      Value *Op0 = CI->getArgOperand(0);
      unsigned Imm = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      unsigned DstNumElts =
          cast<FixedVectorType>(CI->getType())->getNumElements();
      unsigned SrcNumElts =
          cast<FixedVectorType>(Op0->getType())->getNumElements();
      unsigned Scale = SrcNumElts / DstNumElts;

      // Mask off the high bits of the immediate value; hardware ignores those.
      Imm = Imm % Scale;

      // Get indexes for the subvector of the input vector.
      SmallVector<int, 8> Idxs(DstNumElts);
      for (unsigned i = 0; i != DstNumElts; ++i) {
        Idxs[i] = i + (Imm * DstNumElts);
      }
      Rep = Builder.CreateShuffleVector(Op0, Op0, Idxs);

      // If the intrinsic has a mask operand, handle that.
      if (CI->arg_size() == 4)
        Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                            CI->getArgOperand(2));
    } else if (!IsX86 && Name == "stackprotectorcheck") {
      Rep = nullptr;
    } else if (IsX86 && (Name.startswith("avx512.mask.perm.df.") ||
                         Name.startswith("avx512.mask.perm.di."))) {
      Value *Op0 = CI->getArgOperand(0);
      unsigned Imm = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      auto *VecTy = cast<FixedVectorType>(CI->getType());
      unsigned NumElts = VecTy->getNumElements();

      SmallVector<int, 8> Idxs(NumElts);
      for (unsigned i = 0; i != NumElts; ++i)
        Idxs[i] = (i & ~0x3) + ((Imm >> (2 * (i & 0x3))) & 3);

      Rep = Builder.CreateShuffleVector(Op0, Op0, Idxs);

      if (CI->arg_size() == 4)
        Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                            CI->getArgOperand(2));
    } else if (IsX86 && (Name.startswith("avx.vperm2f128.") ||
                         Name == "avx2.vperm2i128")) {
      // The immediate permute control byte looks like this:
      //    [1:0] - select 128 bits from sources for low half of destination
      //    [2]   - ignore
      //    [3]   - zero low half of destination
      //    [5:4] - select 128 bits from sources for high half of destination
      //    [6]   - ignore
      //    [7]   - zero high half of destination

      uint8_t Imm = cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue();

      unsigned NumElts = cast<FixedVectorType>(CI->getType())->getNumElements();
      unsigned HalfSize = NumElts / 2;
      SmallVector<int, 8> ShuffleMask(NumElts);

      // Determine which operand(s) are actually in use for this instruction.
      Value *V0 = (Imm & 0x02) ? CI->getArgOperand(1) : CI->getArgOperand(0);
      Value *V1 = (Imm & 0x20) ? CI->getArgOperand(1) : CI->getArgOperand(0);

      // If needed, replace operands based on zero mask.
      V0 = (Imm & 0x08) ? ConstantAggregateZero::get(CI->getType()) : V0;
      V1 = (Imm & 0x80) ? ConstantAggregateZero::get(CI->getType()) : V1;

      // Permute low half of result.
      unsigned StartIndex = (Imm & 0x01) ? HalfSize : 0;
      for (unsigned i = 0; i < HalfSize; ++i)
        ShuffleMask[i] = StartIndex + i;

      // Permute high half of result.
      StartIndex = (Imm & 0x10) ? HalfSize : 0;
      for (unsigned i = 0; i < HalfSize; ++i)
        ShuffleMask[i + HalfSize] = NumElts + StartIndex + i;

      Rep = Builder.CreateShuffleVector(V0, V1, ShuffleMask);

    } else if (IsX86 && (Name.startswith("avx.vpermil.") ||
                         Name == "sse2.pshuf.d" ||
                         Name.startswith("avx512.mask.vpermil.p") ||
                         Name.startswith("avx512.mask.pshuf.d."))) {
      Value *Op0 = CI->getArgOperand(0);
      unsigned Imm = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      auto *VecTy = cast<FixedVectorType>(CI->getType());
      unsigned NumElts = VecTy->getNumElements();
      // Calculate the size of each index in the immediate.
      unsigned IdxSize = 64 / VecTy->getScalarSizeInBits();
      unsigned IdxMask = ((1 << IdxSize) - 1);

      SmallVector<int, 8> Idxs(NumElts);
      // Lookup the bits for this element, wrapping around the immediate every
      // 8-bits. Elements are grouped into sets of 2 or 4 elements so we need
      // to offset by the first index of each group.
      for (unsigned i = 0; i != NumElts; ++i)
        Idxs[i] = ((Imm >> ((i * IdxSize) % 8)) & IdxMask) | (i & ~IdxMask);

      Rep = Builder.CreateShuffleVector(Op0, Op0, Idxs);

      if (CI->arg_size() == 4)
        Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                            CI->getArgOperand(2));
    } else if (IsX86 && (Name == "sse2.pshufl.w" ||
                         Name.startswith("avx512.mask.pshufl.w."))) {
      Value *Op0 = CI->getArgOperand(0);
      unsigned Imm = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      unsigned NumElts = cast<FixedVectorType>(CI->getType())->getNumElements();

      SmallVector<int, 16> Idxs(NumElts);
      for (unsigned l = 0; l != NumElts; l += 8) {
        for (unsigned i = 0; i != 4; ++i)
          Idxs[i + l] = ((Imm >> (2 * i)) & 0x3) + l;
        for (unsigned i = 4; i != 8; ++i)
          Idxs[i + l] = i + l;
      }

      Rep = Builder.CreateShuffleVector(Op0, Op0, Idxs);

      if (CI->arg_size() == 4)
        Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                            CI->getArgOperand(2));
    } else if (IsX86 && (Name == "sse2.pshufh.w" ||
                         Name.startswith("avx512.mask.pshufh.w."))) {
      Value *Op0 = CI->getArgOperand(0);
      unsigned Imm = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      unsigned NumElts = cast<FixedVectorType>(CI->getType())->getNumElements();

      SmallVector<int, 16> Idxs(NumElts);
      for (unsigned l = 0; l != NumElts; l += 8) {
        for (unsigned i = 0; i != 4; ++i)
          Idxs[i + l] = i + l;
        for (unsigned i = 0; i != 4; ++i)
          Idxs[i + l + 4] = ((Imm >> (2 * i)) & 0x3) + 4 + l;
      }

      Rep = Builder.CreateShuffleVector(Op0, Op0, Idxs);

      if (CI->arg_size() == 4)
        Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                            CI->getArgOperand(2));
    } else if (IsX86 && Name.startswith("avx512.mask.shuf.p")) {
      Value *Op0 = CI->getArgOperand(0);
      Value *Op1 = CI->getArgOperand(1);
      unsigned Imm = cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue();
      unsigned NumElts = cast<FixedVectorType>(CI->getType())->getNumElements();

      unsigned NumLaneElts = 128/CI->getType()->getScalarSizeInBits();
      unsigned HalfLaneElts = NumLaneElts / 2;

      SmallVector<int, 16> Idxs(NumElts);
      for (unsigned i = 0; i != NumElts; ++i) {
        // Base index is the starting element of the lane.
        Idxs[i] = i - (i % NumLaneElts);
        // If we are half way through the lane switch to the other source.
        if ((i % NumLaneElts) >= HalfLaneElts)
          Idxs[i] += NumElts;
        // Now select the specific element. By adding HalfLaneElts bits from
        // the immediate. Wrapping around the immediate every 8-bits.
        Idxs[i] += (Imm >> ((i * HalfLaneElts) % 8)) & ((1 << HalfLaneElts) - 1);
      }

      Rep = Builder.CreateShuffleVector(Op0, Op1, Idxs);

      Rep = EmitX86Select(Builder, CI->getArgOperand(4), Rep,
                          CI->getArgOperand(3));
    } else if (IsX86 && (Name.startswith("avx512.mask.movddup") ||
                         Name.startswith("avx512.mask.movshdup") ||
                         Name.startswith("avx512.mask.movsldup"))) {
      Value *Op0 = CI->getArgOperand(0);
      unsigned NumElts = cast<FixedVectorType>(CI->getType())->getNumElements();
      unsigned NumLaneElts = 128/CI->getType()->getScalarSizeInBits();

      unsigned Offset = 0;
      if (Name.startswith("avx512.mask.movshdup."))
        Offset = 1;

      SmallVector<int, 16> Idxs(NumElts);
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
      int NumElts = cast<FixedVectorType>(CI->getType())->getNumElements();
      int NumLaneElts = 128/CI->getType()->getScalarSizeInBits();

      SmallVector<int, 64> Idxs(NumElts);
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
      int NumElts = cast<FixedVectorType>(CI->getType())->getNumElements();
      int NumLaneElts = 128/CI->getType()->getScalarSizeInBits();

      SmallVector<int, 64> Idxs(NumElts);
      for (int l = 0; l != NumElts; l += NumLaneElts)
        for (int i = 0; i != NumLaneElts; ++i)
          Idxs[i + l] = (NumLaneElts / 2) + l + (i / 2) + NumElts * (i % 2);

      Rep = Builder.CreateShuffleVector(Op0, Op1, Idxs);

      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                          CI->getArgOperand(2));
    } else if (IsX86 && (Name.startswith("avx512.mask.and.") ||
                         Name.startswith("avx512.mask.pand."))) {
      VectorType *FTy = cast<VectorType>(CI->getType());
      VectorType *ITy = VectorType::getInteger(FTy);
      Rep = Builder.CreateAnd(Builder.CreateBitCast(CI->getArgOperand(0), ITy),
                              Builder.CreateBitCast(CI->getArgOperand(1), ITy));
      Rep = Builder.CreateBitCast(Rep, FTy);
      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                          CI->getArgOperand(2));
    } else if (IsX86 && (Name.startswith("avx512.mask.andn.") ||
                         Name.startswith("avx512.mask.pandn."))) {
      VectorType *FTy = cast<VectorType>(CI->getType());
      VectorType *ITy = VectorType::getInteger(FTy);
      Rep = Builder.CreateNot(Builder.CreateBitCast(CI->getArgOperand(0), ITy));
      Rep = Builder.CreateAnd(Rep,
                              Builder.CreateBitCast(CI->getArgOperand(1), ITy));
      Rep = Builder.CreateBitCast(Rep, FTy);
      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                          CI->getArgOperand(2));
    } else if (IsX86 && (Name.startswith("avx512.mask.or.") ||
                         Name.startswith("avx512.mask.por."))) {
      VectorType *FTy = cast<VectorType>(CI->getType());
      VectorType *ITy = VectorType::getInteger(FTy);
      Rep = Builder.CreateOr(Builder.CreateBitCast(CI->getArgOperand(0), ITy),
                             Builder.CreateBitCast(CI->getArgOperand(1), ITy));
      Rep = Builder.CreateBitCast(Rep, FTy);
      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                          CI->getArgOperand(2));
    } else if (IsX86 && (Name.startswith("avx512.mask.xor.") ||
                         Name.startswith("avx512.mask.pxor."))) {
      VectorType *FTy = cast<VectorType>(CI->getType());
      VectorType *ITy = VectorType::getInteger(FTy);
      Rep = Builder.CreateXor(Builder.CreateBitCast(CI->getArgOperand(0), ITy),
                              Builder.CreateBitCast(CI->getArgOperand(1), ITy));
      Rep = Builder.CreateBitCast(Rep, FTy);
      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                          CI->getArgOperand(2));
    } else if (IsX86 && Name.startswith("avx512.mask.padd.")) {
      Rep = Builder.CreateAdd(CI->getArgOperand(0), CI->getArgOperand(1));
      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                          CI->getArgOperand(2));
    } else if (IsX86 && Name.startswith("avx512.mask.psub.")) {
      Rep = Builder.CreateSub(CI->getArgOperand(0), CI->getArgOperand(1));
      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                          CI->getArgOperand(2));
    } else if (IsX86 && Name.startswith("avx512.mask.pmull.")) {
      Rep = Builder.CreateMul(CI->getArgOperand(0), CI->getArgOperand(1));
      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                          CI->getArgOperand(2));
    } else if (IsX86 && Name.startswith("avx512.mask.add.p")) {
      if (Name.endswith(".512")) {
        Intrinsic::ID IID;
        if (Name[17] == 's')
          IID = Intrinsic::x86_avx512_add_ps_512;
        else
          IID = Intrinsic::x86_avx512_add_pd_512;

        Rep = Builder.CreateCall(Intrinsic::getDeclaration(F->getParent(), IID),
                                 { CI->getArgOperand(0), CI->getArgOperand(1),
                                   CI->getArgOperand(4) });
      } else {
        Rep = Builder.CreateFAdd(CI->getArgOperand(0), CI->getArgOperand(1));
      }
      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                          CI->getArgOperand(2));
    } else if (IsX86 && Name.startswith("avx512.mask.div.p")) {
      if (Name.endswith(".512")) {
        Intrinsic::ID IID;
        if (Name[17] == 's')
          IID = Intrinsic::x86_avx512_div_ps_512;
        else
          IID = Intrinsic::x86_avx512_div_pd_512;

        Rep = Builder.CreateCall(Intrinsic::getDeclaration(F->getParent(), IID),
                                 { CI->getArgOperand(0), CI->getArgOperand(1),
                                   CI->getArgOperand(4) });
      } else {
        Rep = Builder.CreateFDiv(CI->getArgOperand(0), CI->getArgOperand(1));
      }
      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                          CI->getArgOperand(2));
    } else if (IsX86 && Name.startswith("avx512.mask.mul.p")) {
      if (Name.endswith(".512")) {
        Intrinsic::ID IID;
        if (Name[17] == 's')
          IID = Intrinsic::x86_avx512_mul_ps_512;
        else
          IID = Intrinsic::x86_avx512_mul_pd_512;

        Rep = Builder.CreateCall(Intrinsic::getDeclaration(F->getParent(), IID),
                                 { CI->getArgOperand(0), CI->getArgOperand(1),
                                   CI->getArgOperand(4) });
      } else {
        Rep = Builder.CreateFMul(CI->getArgOperand(0), CI->getArgOperand(1));
      }
      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                          CI->getArgOperand(2));
    } else if (IsX86 && Name.startswith("avx512.mask.sub.p")) {
      if (Name.endswith(".512")) {
        Intrinsic::ID IID;
        if (Name[17] == 's')
          IID = Intrinsic::x86_avx512_sub_ps_512;
        else
          IID = Intrinsic::x86_avx512_sub_pd_512;

        Rep = Builder.CreateCall(Intrinsic::getDeclaration(F->getParent(), IID),
                                 { CI->getArgOperand(0), CI->getArgOperand(1),
                                   CI->getArgOperand(4) });
      } else {
        Rep = Builder.CreateFSub(CI->getArgOperand(0), CI->getArgOperand(1));
      }
      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                          CI->getArgOperand(2));
    } else if (IsX86 && (Name.startswith("avx512.mask.max.p") ||
                         Name.startswith("avx512.mask.min.p")) &&
               Name.drop_front(18) == ".512") {
      bool IsDouble = Name[17] == 'd';
      bool IsMin = Name[13] == 'i';
      static const Intrinsic::ID MinMaxTbl[2][2] = {
        { Intrinsic::x86_avx512_max_ps_512, Intrinsic::x86_avx512_max_pd_512 },
        { Intrinsic::x86_avx512_min_ps_512, Intrinsic::x86_avx512_min_pd_512 }
      };
      Intrinsic::ID IID = MinMaxTbl[IsMin][IsDouble];

      Rep = Builder.CreateCall(Intrinsic::getDeclaration(F->getParent(), IID),
                               { CI->getArgOperand(0), CI->getArgOperand(1),
                                 CI->getArgOperand(4) });
      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep,
                          CI->getArgOperand(2));
    } else if (IsX86 && Name.startswith("avx512.mask.lzcnt.")) {
      Rep = Builder.CreateCall(Intrinsic::getDeclaration(F->getParent(),
                                                         Intrinsic::ctlz,
                                                         CI->getType()),
                               { CI->getArgOperand(0), Builder.getInt1(false) });
      Rep = EmitX86Select(Builder, CI->getArgOperand(2), Rep,
                          CI->getArgOperand(1));
    } else if (IsX86 && Name.startswith("avx512.mask.psll")) {
      bool IsImmediate = Name[16] == 'i' ||
                         (Name.size() > 18 && Name[18] == 'i');
      bool IsVariable = Name[16] == 'v';
      char Size = Name[16] == '.' ? Name[17] :
                  Name[17] == '.' ? Name[18] :
                  Name[18] == '.' ? Name[19] :
                                    Name[20];

      Intrinsic::ID IID;
      if (IsVariable && Name[17] != '.') {
        if (Size == 'd' && Name[17] == '2') // avx512.mask.psllv2.di
          IID = Intrinsic::x86_avx2_psllv_q;
        else if (Size == 'd' && Name[17] == '4') // avx512.mask.psllv4.di
          IID = Intrinsic::x86_avx2_psllv_q_256;
        else if (Size == 's' && Name[17] == '4') // avx512.mask.psllv4.si
          IID = Intrinsic::x86_avx2_psllv_d;
        else if (Size == 's' && Name[17] == '8') // avx512.mask.psllv8.si
          IID = Intrinsic::x86_avx2_psllv_d_256;
        else if (Size == 'h' && Name[17] == '8') // avx512.mask.psllv8.hi
          IID = Intrinsic::x86_avx512_psllv_w_128;
        else if (Size == 'h' && Name[17] == '1') // avx512.mask.psllv16.hi
          IID = Intrinsic::x86_avx512_psllv_w_256;
        else if (Name[17] == '3' && Name[18] == '2') // avx512.mask.psllv32hi
          IID = Intrinsic::x86_avx512_psllv_w_512;
        else
          llvm_unreachable("Unexpected size");
      } else if (Name.endswith(".128")) {
        if (Size == 'd') // avx512.mask.psll.d.128, avx512.mask.psll.di.128
          IID = IsImmediate ? Intrinsic::x86_sse2_pslli_d
                            : Intrinsic::x86_sse2_psll_d;
        else if (Size == 'q') // avx512.mask.psll.q.128, avx512.mask.psll.qi.128
          IID = IsImmediate ? Intrinsic::x86_sse2_pslli_q
                            : Intrinsic::x86_sse2_psll_q;
        else if (Size == 'w') // avx512.mask.psll.w.128, avx512.mask.psll.wi.128
          IID = IsImmediate ? Intrinsic::x86_sse2_pslli_w
                            : Intrinsic::x86_sse2_psll_w;
        else
          llvm_unreachable("Unexpected size");
      } else if (Name.endswith(".256")) {
        if (Size == 'd') // avx512.mask.psll.d.256, avx512.mask.psll.di.256
          IID = IsImmediate ? Intrinsic::x86_avx2_pslli_d
                            : Intrinsic::x86_avx2_psll_d;
        else if (Size == 'q') // avx512.mask.psll.q.256, avx512.mask.psll.qi.256
          IID = IsImmediate ? Intrinsic::x86_avx2_pslli_q
                            : Intrinsic::x86_avx2_psll_q;
        else if (Size == 'w') // avx512.mask.psll.w.256, avx512.mask.psll.wi.256
          IID = IsImmediate ? Intrinsic::x86_avx2_pslli_w
                            : Intrinsic::x86_avx2_psll_w;
        else
          llvm_unreachable("Unexpected size");
      } else {
        if (Size == 'd') // psll.di.512, pslli.d, psll.d, psllv.d.512
          IID = IsImmediate ? Intrinsic::x86_avx512_pslli_d_512 :
                IsVariable  ? Intrinsic::x86_avx512_psllv_d_512 :
                              Intrinsic::x86_avx512_psll_d_512;
        else if (Size == 'q') // psll.qi.512, pslli.q, psll.q, psllv.q.512
          IID = IsImmediate ? Intrinsic::x86_avx512_pslli_q_512 :
                IsVariable  ? Intrinsic::x86_avx512_psllv_q_512 :
                              Intrinsic::x86_avx512_psll_q_512;
        else if (Size == 'w') // psll.wi.512, pslli.w, psll.w
          IID = IsImmediate ? Intrinsic::x86_avx512_pslli_w_512
                            : Intrinsic::x86_avx512_psll_w_512;
        else
          llvm_unreachable("Unexpected size");
      }

      Rep = UpgradeX86MaskedShift(Builder, *CI, IID);
    } else if (IsX86 && Name.startswith("avx512.mask.psrl")) {
      bool IsImmediate = Name[16] == 'i' ||
                         (Name.size() > 18 && Name[18] == 'i');
      bool IsVariable = Name[16] == 'v';
      char Size = Name[16] == '.' ? Name[17] :
                  Name[17] == '.' ? Name[18] :
                  Name[18] == '.' ? Name[19] :
                                    Name[20];

      Intrinsic::ID IID;
      if (IsVariable && Name[17] != '.') {
        if (Size == 'd' && Name[17] == '2') // avx512.mask.psrlv2.di
          IID = Intrinsic::x86_avx2_psrlv_q;
        else if (Size == 'd' && Name[17] == '4') // avx512.mask.psrlv4.di
          IID = Intrinsic::x86_avx2_psrlv_q_256;
        else if (Size == 's' && Name[17] == '4') // avx512.mask.psrlv4.si
          IID = Intrinsic::x86_avx2_psrlv_d;
        else if (Size == 's' && Name[17] == '8') // avx512.mask.psrlv8.si
          IID = Intrinsic::x86_avx2_psrlv_d_256;
        else if (Size == 'h' && Name[17] == '8') // avx512.mask.psrlv8.hi
          IID = Intrinsic::x86_avx512_psrlv_w_128;
        else if (Size == 'h' && Name[17] == '1') // avx512.mask.psrlv16.hi
          IID = Intrinsic::x86_avx512_psrlv_w_256;
        else if (Name[17] == '3' && Name[18] == '2') // avx512.mask.psrlv32hi
          IID = Intrinsic::x86_avx512_psrlv_w_512;
        else
          llvm_unreachable("Unexpected size");
      } else if (Name.endswith(".128")) {
        if (Size == 'd') // avx512.mask.psrl.d.128, avx512.mask.psrl.di.128
          IID = IsImmediate ? Intrinsic::x86_sse2_psrli_d
                            : Intrinsic::x86_sse2_psrl_d;
        else if (Size == 'q') // avx512.mask.psrl.q.128, avx512.mask.psrl.qi.128
          IID = IsImmediate ? Intrinsic::x86_sse2_psrli_q
                            : Intrinsic::x86_sse2_psrl_q;
        else if (Size == 'w') // avx512.mask.psrl.w.128, avx512.mask.psrl.wi.128
          IID = IsImmediate ? Intrinsic::x86_sse2_psrli_w
                            : Intrinsic::x86_sse2_psrl_w;
        else
          llvm_unreachable("Unexpected size");
      } else if (Name.endswith(".256")) {
        if (Size == 'd') // avx512.mask.psrl.d.256, avx512.mask.psrl.di.256
          IID = IsImmediate ? Intrinsic::x86_avx2_psrli_d
                            : Intrinsic::x86_avx2_psrl_d;
        else if (Size == 'q') // avx512.mask.psrl.q.256, avx512.mask.psrl.qi.256
          IID = IsImmediate ? Intrinsic::x86_avx2_psrli_q
                            : Intrinsic::x86_avx2_psrl_q;
        else if (Size == 'w') // avx512.mask.psrl.w.256, avx512.mask.psrl.wi.256
          IID = IsImmediate ? Intrinsic::x86_avx2_psrli_w
                            : Intrinsic::x86_avx2_psrl_w;
        else
          llvm_unreachable("Unexpected size");
      } else {
        if (Size == 'd') // psrl.di.512, psrli.d, psrl.d, psrl.d.512
          IID = IsImmediate ? Intrinsic::x86_avx512_psrli_d_512 :
                IsVariable  ? Intrinsic::x86_avx512_psrlv_d_512 :
                              Intrinsic::x86_avx512_psrl_d_512;
        else if (Size == 'q') // psrl.qi.512, psrli.q, psrl.q, psrl.q.512
          IID = IsImmediate ? Intrinsic::x86_avx512_psrli_q_512 :
                IsVariable  ? Intrinsic::x86_avx512_psrlv_q_512 :
                              Intrinsic::x86_avx512_psrl_q_512;
        else if (Size == 'w') // psrl.wi.512, psrli.w, psrl.w)
          IID = IsImmediate ? Intrinsic::x86_avx512_psrli_w_512
                            : Intrinsic::x86_avx512_psrl_w_512;
        else
          llvm_unreachable("Unexpected size");
      }

      Rep = UpgradeX86MaskedShift(Builder, *CI, IID);
    } else if (IsX86 && Name.startswith("avx512.mask.psra")) {
      bool IsImmediate = Name[16] == 'i' ||
                         (Name.size() > 18 && Name[18] == 'i');
      bool IsVariable = Name[16] == 'v';
      char Size = Name[16] == '.' ? Name[17] :
                  Name[17] == '.' ? Name[18] :
                  Name[18] == '.' ? Name[19] :
                                    Name[20];

      Intrinsic::ID IID;
      if (IsVariable && Name[17] != '.') {
        if (Size == 's' && Name[17] == '4') // avx512.mask.psrav4.si
          IID = Intrinsic::x86_avx2_psrav_d;
        else if (Size == 's' && Name[17] == '8') // avx512.mask.psrav8.si
          IID = Intrinsic::x86_avx2_psrav_d_256;
        else if (Size == 'h' && Name[17] == '8') // avx512.mask.psrav8.hi
          IID = Intrinsic::x86_avx512_psrav_w_128;
        else if (Size == 'h' && Name[17] == '1') // avx512.mask.psrav16.hi
          IID = Intrinsic::x86_avx512_psrav_w_256;
        else if (Name[17] == '3' && Name[18] == '2') // avx512.mask.psrav32hi
          IID = Intrinsic::x86_avx512_psrav_w_512;
        else
          llvm_unreachable("Unexpected size");
      } else if (Name.endswith(".128")) {
        if (Size == 'd') // avx512.mask.psra.d.128, avx512.mask.psra.di.128
          IID = IsImmediate ? Intrinsic::x86_sse2_psrai_d
                            : Intrinsic::x86_sse2_psra_d;
        else if (Size == 'q') // avx512.mask.psra.q.128, avx512.mask.psra.qi.128
          IID = IsImmediate ? Intrinsic::x86_avx512_psrai_q_128 :
                IsVariable  ? Intrinsic::x86_avx512_psrav_q_128 :
                              Intrinsic::x86_avx512_psra_q_128;
        else if (Size == 'w') // avx512.mask.psra.w.128, avx512.mask.psra.wi.128
          IID = IsImmediate ? Intrinsic::x86_sse2_psrai_w
                            : Intrinsic::x86_sse2_psra_w;
        else
          llvm_unreachable("Unexpected size");
      } else if (Name.endswith(".256")) {
        if (Size == 'd') // avx512.mask.psra.d.256, avx512.mask.psra.di.256
          IID = IsImmediate ? Intrinsic::x86_avx2_psrai_d
                            : Intrinsic::x86_avx2_psra_d;
        else if (Size == 'q') // avx512.mask.psra.q.256, avx512.mask.psra.qi.256
          IID = IsImmediate ? Intrinsic::x86_avx512_psrai_q_256 :
                IsVariable  ? Intrinsic::x86_avx512_psrav_q_256 :
                              Intrinsic::x86_avx512_psra_q_256;
        else if (Size == 'w') // avx512.mask.psra.w.256, avx512.mask.psra.wi.256
          IID = IsImmediate ? Intrinsic::x86_avx2_psrai_w
                            : Intrinsic::x86_avx2_psra_w;
        else
          llvm_unreachable("Unexpected size");
      } else {
        if (Size == 'd') // psra.di.512, psrai.d, psra.d, psrav.d.512
          IID = IsImmediate ? Intrinsic::x86_avx512_psrai_d_512 :
                IsVariable  ? Intrinsic::x86_avx512_psrav_d_512 :
                              Intrinsic::x86_avx512_psra_d_512;
        else if (Size == 'q') // psra.qi.512, psrai.q, psra.q
          IID = IsImmediate ? Intrinsic::x86_avx512_psrai_q_512 :
                IsVariable  ? Intrinsic::x86_avx512_psrav_q_512 :
                              Intrinsic::x86_avx512_psra_q_512;
        else if (Size == 'w') // psra.wi.512, psrai.w, psra.w
          IID = IsImmediate ? Intrinsic::x86_avx512_psrai_w_512
                            : Intrinsic::x86_avx512_psra_w_512;
        else
          llvm_unreachable("Unexpected size");
      }

      Rep = UpgradeX86MaskedShift(Builder, *CI, IID);
    } else if (IsX86 && Name.startswith("avx512.mask.move.s")) {
      Rep = upgradeMaskedMove(Builder, *CI);
    } else if (IsX86 && Name.startswith("avx512.cvtmask2")) {
      Rep = UpgradeMaskToInt(Builder, *CI);
    } else if (IsX86 && Name.endswith(".movntdqa")) {
      MDNode *Node = MDNode::get(
          C, ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(C), 1)));

      Value *Ptr = CI->getArgOperand(0);

      // Convert the type of the pointer to a pointer to the stored type.
      Value *BC = Builder.CreateBitCast(
          Ptr, PointerType::getUnqual(CI->getType()), "cast");
      LoadInst *LI = Builder.CreateAlignedLoad(
          CI->getType(), BC,
          Align(CI->getType()->getPrimitiveSizeInBits().getFixedValue() / 8));
      LI->setMetadata(LLVMContext::MD_nontemporal, Node);
      Rep = LI;
    } else if (IsX86 && (Name.startswith("fma.vfmadd.") ||
                         Name.startswith("fma.vfmsub.") ||
                         Name.startswith("fma.vfnmadd.") ||
                         Name.startswith("fma.vfnmsub."))) {
      bool NegMul = Name[6] == 'n';
      bool NegAcc = NegMul ? Name[8] == 's' : Name[7] == 's';
      bool IsScalar = NegMul ? Name[12] == 's' : Name[11] == 's';

      Value *Ops[] = { CI->getArgOperand(0), CI->getArgOperand(1),
                       CI->getArgOperand(2) };

      if (IsScalar) {
        Ops[0] = Builder.CreateExtractElement(Ops[0], (uint64_t)0);
        Ops[1] = Builder.CreateExtractElement(Ops[1], (uint64_t)0);
        Ops[2] = Builder.CreateExtractElement(Ops[2], (uint64_t)0);
      }

      if (NegMul && !IsScalar)
        Ops[0] = Builder.CreateFNeg(Ops[0]);
      if (NegMul && IsScalar)
        Ops[1] = Builder.CreateFNeg(Ops[1]);
      if (NegAcc)
        Ops[2] = Builder.CreateFNeg(Ops[2]);

      Rep = Builder.CreateCall(Intrinsic::getDeclaration(CI->getModule(),
                                                         Intrinsic::fma,
                                                         Ops[0]->getType()),
                               Ops);

      if (IsScalar)
        Rep = Builder.CreateInsertElement(CI->getArgOperand(0), Rep,
                                          (uint64_t)0);
    } else if (IsX86 && Name.startswith("fma4.vfmadd.s")) {
      Value *Ops[] = { CI->getArgOperand(0), CI->getArgOperand(1),
                       CI->getArgOperand(2) };

      Ops[0] = Builder.CreateExtractElement(Ops[0], (uint64_t)0);
      Ops[1] = Builder.CreateExtractElement(Ops[1], (uint64_t)0);
      Ops[2] = Builder.CreateExtractElement(Ops[2], (uint64_t)0);

      Rep = Builder.CreateCall(Intrinsic::getDeclaration(CI->getModule(),
                                                         Intrinsic::fma,
                                                         Ops[0]->getType()),
                               Ops);

      Rep = Builder.CreateInsertElement(Constant::getNullValue(CI->getType()),
                                        Rep, (uint64_t)0);
    } else if (IsX86 && (Name.startswith("avx512.mask.vfmadd.s") ||
                         Name.startswith("avx512.maskz.vfmadd.s") ||
                         Name.startswith("avx512.mask3.vfmadd.s") ||
                         Name.startswith("avx512.mask3.vfmsub.s") ||
                         Name.startswith("avx512.mask3.vfnmsub.s"))) {
      bool IsMask3 = Name[11] == '3';
      bool IsMaskZ = Name[11] == 'z';
      // Drop the "avx512.mask." to make it easier.
      Name = Name.drop_front(IsMask3 || IsMaskZ ? 13 : 12);
      bool NegMul = Name[2] == 'n';
      bool NegAcc = NegMul ? Name[4] == 's' : Name[3] == 's';

      Value *A = CI->getArgOperand(0);
      Value *B = CI->getArgOperand(1);
      Value *C = CI->getArgOperand(2);

      if (NegMul && (IsMask3 || IsMaskZ))
        A = Builder.CreateFNeg(A);
      if (NegMul && !(IsMask3 || IsMaskZ))
        B = Builder.CreateFNeg(B);
      if (NegAcc)
        C = Builder.CreateFNeg(C);

      A = Builder.CreateExtractElement(A, (uint64_t)0);
      B = Builder.CreateExtractElement(B, (uint64_t)0);
      C = Builder.CreateExtractElement(C, (uint64_t)0);

      if (!isa<ConstantInt>(CI->getArgOperand(4)) ||
          cast<ConstantInt>(CI->getArgOperand(4))->getZExtValue() != 4) {
        Value *Ops[] = { A, B, C, CI->getArgOperand(4) };

        Intrinsic::ID IID;
        if (Name.back() == 'd')
          IID = Intrinsic::x86_avx512_vfmadd_f64;
        else
          IID = Intrinsic::x86_avx512_vfmadd_f32;
        Function *FMA = Intrinsic::getDeclaration(CI->getModule(), IID);
        Rep = Builder.CreateCall(FMA, Ops);
      } else {
        Function *FMA = Intrinsic::getDeclaration(CI->getModule(),
                                                  Intrinsic::fma,
                                                  A->getType());
        Rep = Builder.CreateCall(FMA, { A, B, C });
      }

      Value *PassThru = IsMaskZ ? Constant::getNullValue(Rep->getType()) :
                        IsMask3 ? C : A;

      // For Mask3 with NegAcc, we need to create a new extractelement that
      // avoids the negation above.
      if (NegAcc && IsMask3)
        PassThru = Builder.CreateExtractElement(CI->getArgOperand(2),
                                                (uint64_t)0);

      Rep = EmitX86ScalarSelect(Builder, CI->getArgOperand(3),
                                Rep, PassThru);
      Rep = Builder.CreateInsertElement(CI->getArgOperand(IsMask3 ? 2 : 0),
                                        Rep, (uint64_t)0);
    } else if (IsX86 && (Name.startswith("avx512.mask.vfmadd.p") ||
                         Name.startswith("avx512.mask.vfnmadd.p") ||
                         Name.startswith("avx512.mask.vfnmsub.p") ||
                         Name.startswith("avx512.mask3.vfmadd.p") ||
                         Name.startswith("avx512.mask3.vfmsub.p") ||
                         Name.startswith("avx512.mask3.vfnmsub.p") ||
                         Name.startswith("avx512.maskz.vfmadd.p"))) {
      bool IsMask3 = Name[11] == '3';
      bool IsMaskZ = Name[11] == 'z';
      // Drop the "avx512.mask." to make it easier.
      Name = Name.drop_front(IsMask3 || IsMaskZ ? 13 : 12);
      bool NegMul = Name[2] == 'n';
      bool NegAcc = NegMul ? Name[4] == 's' : Name[3] == 's';

      Value *A = CI->getArgOperand(0);
      Value *B = CI->getArgOperand(1);
      Value *C = CI->getArgOperand(2);

      if (NegMul && (IsMask3 || IsMaskZ))
        A = Builder.CreateFNeg(A);
      if (NegMul && !(IsMask3 || IsMaskZ))
        B = Builder.CreateFNeg(B);
      if (NegAcc)
        C = Builder.CreateFNeg(C);

      if (CI->arg_size() == 5 &&
          (!isa<ConstantInt>(CI->getArgOperand(4)) ||
           cast<ConstantInt>(CI->getArgOperand(4))->getZExtValue() != 4)) {
        Intrinsic::ID IID;
        // Check the character before ".512" in string.
        if (Name[Name.size()-5] == 's')
          IID = Intrinsic::x86_avx512_vfmadd_ps_512;
        else
          IID = Intrinsic::x86_avx512_vfmadd_pd_512;

        Rep = Builder.CreateCall(Intrinsic::getDeclaration(F->getParent(), IID),
                                 { A, B, C, CI->getArgOperand(4) });
      } else {
        Function *FMA = Intrinsic::getDeclaration(CI->getModule(),
                                                  Intrinsic::fma,
                                                  A->getType());
        Rep = Builder.CreateCall(FMA, { A, B, C });
      }

      Value *PassThru = IsMaskZ ? llvm::Constant::getNullValue(CI->getType()) :
                        IsMask3 ? CI->getArgOperand(2) :
                                  CI->getArgOperand(0);

      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep, PassThru);
    } else if (IsX86 &&  Name.startswith("fma.vfmsubadd.p")) {
      unsigned VecWidth = CI->getType()->getPrimitiveSizeInBits();
      unsigned EltWidth = CI->getType()->getScalarSizeInBits();
      Intrinsic::ID IID;
      if (VecWidth == 128 && EltWidth == 32)
        IID = Intrinsic::x86_fma_vfmaddsub_ps;
      else if (VecWidth == 256 && EltWidth == 32)
        IID = Intrinsic::x86_fma_vfmaddsub_ps_256;
      else if (VecWidth == 128 && EltWidth == 64)
        IID = Intrinsic::x86_fma_vfmaddsub_pd;
      else if (VecWidth == 256 && EltWidth == 64)
        IID = Intrinsic::x86_fma_vfmaddsub_pd_256;
      else
        llvm_unreachable("Unexpected intrinsic");

      Value *Ops[] = { CI->getArgOperand(0), CI->getArgOperand(1),
                       CI->getArgOperand(2) };
      Ops[2] = Builder.CreateFNeg(Ops[2]);
      Rep = Builder.CreateCall(Intrinsic::getDeclaration(F->getParent(), IID),
                               Ops);
    } else if (IsX86 && (Name.startswith("avx512.mask.vfmaddsub.p") ||
                         Name.startswith("avx512.mask3.vfmaddsub.p") ||
                         Name.startswith("avx512.maskz.vfmaddsub.p") ||
                         Name.startswith("avx512.mask3.vfmsubadd.p"))) {
      bool IsMask3 = Name[11] == '3';
      bool IsMaskZ = Name[11] == 'z';
      // Drop the "avx512.mask." to make it easier.
      Name = Name.drop_front(IsMask3 || IsMaskZ ? 13 : 12);
      bool IsSubAdd = Name[3] == 's';
      if (CI->arg_size() == 5) {
        Intrinsic::ID IID;
        // Check the character before ".512" in string.
        if (Name[Name.size()-5] == 's')
          IID = Intrinsic::x86_avx512_vfmaddsub_ps_512;
        else
          IID = Intrinsic::x86_avx512_vfmaddsub_pd_512;

        Value *Ops[] = { CI->getArgOperand(0), CI->getArgOperand(1),
                         CI->getArgOperand(2), CI->getArgOperand(4) };
        if (IsSubAdd)
          Ops[2] = Builder.CreateFNeg(Ops[2]);

        Rep = Builder.CreateCall(Intrinsic::getDeclaration(F->getParent(), IID),
                                 Ops);
      } else {
        int NumElts = cast<FixedVectorType>(CI->getType())->getNumElements();

        Value *Ops[] = { CI->getArgOperand(0), CI->getArgOperand(1),
                         CI->getArgOperand(2) };

        Function *FMA = Intrinsic::getDeclaration(CI->getModule(), Intrinsic::fma,
                                                  Ops[0]->getType());
        Value *Odd = Builder.CreateCall(FMA, Ops);
        Ops[2] = Builder.CreateFNeg(Ops[2]);
        Value *Even = Builder.CreateCall(FMA, Ops);

        if (IsSubAdd)
          std::swap(Even, Odd);

        SmallVector<int, 32> Idxs(NumElts);
        for (int i = 0; i != NumElts; ++i)
          Idxs[i] = i + (i % 2) * NumElts;

        Rep = Builder.CreateShuffleVector(Even, Odd, Idxs);
      }

      Value *PassThru = IsMaskZ ? llvm::Constant::getNullValue(CI->getType()) :
                        IsMask3 ? CI->getArgOperand(2) :
                                  CI->getArgOperand(0);

      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep, PassThru);
    } else if (IsX86 && (Name.startswith("avx512.mask.pternlog.") ||
                         Name.startswith("avx512.maskz.pternlog."))) {
      bool ZeroMask = Name[11] == 'z';
      unsigned VecWidth = CI->getType()->getPrimitiveSizeInBits();
      unsigned EltWidth = CI->getType()->getScalarSizeInBits();
      Intrinsic::ID IID;
      if (VecWidth == 128 && EltWidth == 32)
        IID = Intrinsic::x86_avx512_pternlog_d_128;
      else if (VecWidth == 256 && EltWidth == 32)
        IID = Intrinsic::x86_avx512_pternlog_d_256;
      else if (VecWidth == 512 && EltWidth == 32)
        IID = Intrinsic::x86_avx512_pternlog_d_512;
      else if (VecWidth == 128 && EltWidth == 64)
        IID = Intrinsic::x86_avx512_pternlog_q_128;
      else if (VecWidth == 256 && EltWidth == 64)
        IID = Intrinsic::x86_avx512_pternlog_q_256;
      else if (VecWidth == 512 && EltWidth == 64)
        IID = Intrinsic::x86_avx512_pternlog_q_512;
      else
        llvm_unreachable("Unexpected intrinsic");

      Value *Args[] = { CI->getArgOperand(0) , CI->getArgOperand(1),
                        CI->getArgOperand(2), CI->getArgOperand(3) };
      Rep = Builder.CreateCall(Intrinsic::getDeclaration(CI->getModule(), IID),
                               Args);
      Value *PassThru = ZeroMask ? ConstantAggregateZero::get(CI->getType())
                                 : CI->getArgOperand(0);
      Rep = EmitX86Select(Builder, CI->getArgOperand(4), Rep, PassThru);
    } else if (IsX86 && (Name.startswith("avx512.mask.vpmadd52") ||
                         Name.startswith("avx512.maskz.vpmadd52"))) {
      bool ZeroMask = Name[11] == 'z';
      bool High = Name[20] == 'h' || Name[21] == 'h';
      unsigned VecWidth = CI->getType()->getPrimitiveSizeInBits();
      Intrinsic::ID IID;
      if (VecWidth == 128 && !High)
        IID = Intrinsic::x86_avx512_vpmadd52l_uq_128;
      else if (VecWidth == 256 && !High)
        IID = Intrinsic::x86_avx512_vpmadd52l_uq_256;
      else if (VecWidth == 512 && !High)
        IID = Intrinsic::x86_avx512_vpmadd52l_uq_512;
      else if (VecWidth == 128 && High)
        IID = Intrinsic::x86_avx512_vpmadd52h_uq_128;
      else if (VecWidth == 256 && High)
        IID = Intrinsic::x86_avx512_vpmadd52h_uq_256;
      else if (VecWidth == 512 && High)
        IID = Intrinsic::x86_avx512_vpmadd52h_uq_512;
      else
        llvm_unreachable("Unexpected intrinsic");

      Value *Args[] = { CI->getArgOperand(0) , CI->getArgOperand(1),
                        CI->getArgOperand(2) };
      Rep = Builder.CreateCall(Intrinsic::getDeclaration(CI->getModule(), IID),
                               Args);
      Value *PassThru = ZeroMask ? ConstantAggregateZero::get(CI->getType())
                                 : CI->getArgOperand(0);
      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep, PassThru);
    } else if (IsX86 && (Name.startswith("avx512.mask.vpermi2var.") ||
                         Name.startswith("avx512.mask.vpermt2var.") ||
                         Name.startswith("avx512.maskz.vpermt2var."))) {
      bool ZeroMask = Name[11] == 'z';
      bool IndexForm = Name[17] == 'i';
      Rep = UpgradeX86VPERMT2Intrinsics(Builder, *CI, ZeroMask, IndexForm);
    } else if (IsX86 && (Name.startswith("avx512.mask.vpdpbusd.") ||
                         Name.startswith("avx512.maskz.vpdpbusd.") ||
                         Name.startswith("avx512.mask.vpdpbusds.") ||
                         Name.startswith("avx512.maskz.vpdpbusds."))) {
      bool ZeroMask = Name[11] == 'z';
      bool IsSaturating = Name[ZeroMask ? 21 : 20] == 's';
      unsigned VecWidth = CI->getType()->getPrimitiveSizeInBits();
      Intrinsic::ID IID;
      if (VecWidth == 128 && !IsSaturating)
        IID = Intrinsic::x86_avx512_vpdpbusd_128;
      else if (VecWidth == 256 && !IsSaturating)
        IID = Intrinsic::x86_avx512_vpdpbusd_256;
      else if (VecWidth == 512 && !IsSaturating)
        IID = Intrinsic::x86_avx512_vpdpbusd_512;
      else if (VecWidth == 128 && IsSaturating)
        IID = Intrinsic::x86_avx512_vpdpbusds_128;
      else if (VecWidth == 256 && IsSaturating)
        IID = Intrinsic::x86_avx512_vpdpbusds_256;
      else if (VecWidth == 512 && IsSaturating)
        IID = Intrinsic::x86_avx512_vpdpbusds_512;
      else
        llvm_unreachable("Unexpected intrinsic");

      Value *Args[] = { CI->getArgOperand(0), CI->getArgOperand(1),
                        CI->getArgOperand(2)  };
      Rep = Builder.CreateCall(Intrinsic::getDeclaration(CI->getModule(), IID),
                               Args);
      Value *PassThru = ZeroMask ? ConstantAggregateZero::get(CI->getType())
                                 : CI->getArgOperand(0);
      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep, PassThru);
    } else if (IsX86 && (Name.startswith("avx512.mask.vpdpwssd.") ||
                         Name.startswith("avx512.maskz.vpdpwssd.") ||
                         Name.startswith("avx512.mask.vpdpwssds.") ||
                         Name.startswith("avx512.maskz.vpdpwssds."))) {
      bool ZeroMask = Name[11] == 'z';
      bool IsSaturating = Name[ZeroMask ? 21 : 20] == 's';
      unsigned VecWidth = CI->getType()->getPrimitiveSizeInBits();
      Intrinsic::ID IID;
      if (VecWidth == 128 && !IsSaturating)
        IID = Intrinsic::x86_avx512_vpdpwssd_128;
      else if (VecWidth == 256 && !IsSaturating)
        IID = Intrinsic::x86_avx512_vpdpwssd_256;
      else if (VecWidth == 512 && !IsSaturating)
        IID = Intrinsic::x86_avx512_vpdpwssd_512;
      else if (VecWidth == 128 && IsSaturating)
        IID = Intrinsic::x86_avx512_vpdpwssds_128;
      else if (VecWidth == 256 && IsSaturating)
        IID = Intrinsic::x86_avx512_vpdpwssds_256;
      else if (VecWidth == 512 && IsSaturating)
        IID = Intrinsic::x86_avx512_vpdpwssds_512;
      else
        llvm_unreachable("Unexpected intrinsic");

      Value *Args[] = { CI->getArgOperand(0), CI->getArgOperand(1),
                        CI->getArgOperand(2)  };
      Rep = Builder.CreateCall(Intrinsic::getDeclaration(CI->getModule(), IID),
                               Args);
      Value *PassThru = ZeroMask ? ConstantAggregateZero::get(CI->getType())
                                 : CI->getArgOperand(0);
      Rep = EmitX86Select(Builder, CI->getArgOperand(3), Rep, PassThru);
    } else if (IsX86 && (Name == "addcarryx.u32" || Name == "addcarryx.u64" ||
                         Name == "addcarry.u32" || Name == "addcarry.u64" ||
                         Name == "subborrow.u32" || Name == "subborrow.u64")) {
      Intrinsic::ID IID;
      if (Name[0] == 'a' && Name.back() == '2')
        IID = Intrinsic::x86_addcarry_32;
      else if (Name[0] == 'a' && Name.back() == '4')
        IID = Intrinsic::x86_addcarry_64;
      else if (Name[0] == 's' && Name.back() == '2')
        IID = Intrinsic::x86_subborrow_32;
      else if (Name[0] == 's' && Name.back() == '4')
        IID = Intrinsic::x86_subborrow_64;
      else
        llvm_unreachable("Unexpected intrinsic");

      // Make a call with 3 operands.
      Value *Args[] = { CI->getArgOperand(0), CI->getArgOperand(1),
                        CI->getArgOperand(2)};
      Value *NewCall = Builder.CreateCall(
                                Intrinsic::getDeclaration(CI->getModule(), IID),
                                Args);

      // Extract the second result and store it.
      Value *Data = Builder.CreateExtractValue(NewCall, 1);
      // Cast the pointer to the right type.
      Value *Ptr = Builder.CreateBitCast(CI->getArgOperand(3),
                                 llvm::PointerType::getUnqual(Data->getType()));
      Builder.CreateAlignedStore(Data, Ptr, Align(1));
      // Replace the original call result with the first result of the new call.
      Value *CF = Builder.CreateExtractValue(NewCall, 0);

      CI->replaceAllUsesWith(CF);
      Rep = nullptr;
    } else if (IsX86 && Name.startswith("avx512.mask.") &&
               upgradeAVX512MaskToSelect(Name, Builder, *CI, Rep)) {
      // Rep will be updated by the call in the condition.
    } else if (IsNVVM && (Name == "abs.i" || Name == "abs.ll")) {
      Value *Arg = CI->getArgOperand(0);
      Value *Neg = Builder.CreateNeg(Arg, "neg");
      Value *Cmp = Builder.CreateICmpSGE(
          Arg, llvm::Constant::getNullValue(Arg->getType()), "abs.cond");
      Rep = Builder.CreateSelect(Cmp, Arg, Neg, "abs");
    } else if (IsNVVM && (Name.startswith("atomic.load.add.f32.p") ||
                          Name.startswith("atomic.load.add.f64.p"))) {
      Value *Ptr = CI->getArgOperand(0);
      Value *Val = CI->getArgOperand(1);
      Rep = Builder.CreateAtomicRMW(AtomicRMWInst::FAdd, Ptr, Val, MaybeAlign(),
                                    AtomicOrdering::SequentiallyConsistent);
    } else if (IsNVVM && (Name == "max.i" || Name == "max.ll" ||
                          Name == "max.ui" || Name == "max.ull")) {
      Value *Arg0 = CI->getArgOperand(0);
      Value *Arg1 = CI->getArgOperand(1);
      Value *Cmp = Name.endswith(".ui") || Name.endswith(".ull")
                       ? Builder.CreateICmpUGE(Arg0, Arg1, "max.cond")
                       : Builder.CreateICmpSGE(Arg0, Arg1, "max.cond");
      Rep = Builder.CreateSelect(Cmp, Arg0, Arg1, "max");
    } else if (IsNVVM && (Name == "min.i" || Name == "min.ll" ||
                          Name == "min.ui" || Name == "min.ull")) {
      Value *Arg0 = CI->getArgOperand(0);
      Value *Arg1 = CI->getArgOperand(1);
      Value *Cmp = Name.endswith(".ui") || Name.endswith(".ull")
                       ? Builder.CreateICmpULE(Arg0, Arg1, "min.cond")
                       : Builder.CreateICmpSLE(Arg0, Arg1, "min.cond");
      Rep = Builder.CreateSelect(Cmp, Arg0, Arg1, "min");
    } else if (IsNVVM && Name == "clz.ll") {
      // llvm.nvvm.clz.ll returns an i32, but llvm.ctlz.i64 and returns an i64.
      Value *Arg = CI->getArgOperand(0);
      Value *Ctlz = Builder.CreateCall(
          Intrinsic::getDeclaration(F->getParent(), Intrinsic::ctlz,
                                    {Arg->getType()}),
          {Arg, Builder.getFalse()}, "ctlz");
      Rep = Builder.CreateTrunc(Ctlz, Builder.getInt32Ty(), "ctlz.trunc");
    } else if (IsNVVM && Name == "popc.ll") {
      // llvm.nvvm.popc.ll returns an i32, but llvm.ctpop.i64 and returns an
      // i64.
      Value *Arg = CI->getArgOperand(0);
      Value *Popc = Builder.CreateCall(
          Intrinsic::getDeclaration(F->getParent(), Intrinsic::ctpop,
                                    {Arg->getType()}),
          Arg, "ctpop");
      Rep = Builder.CreateTrunc(Popc, Builder.getInt32Ty(), "ctpop.trunc");
    } else if (IsNVVM) {
      if (Name == "h2f") {
        Rep =
            Builder.CreateCall(Intrinsic::getDeclaration(
                                   F->getParent(), Intrinsic::convert_from_fp16,
                                   {Builder.getFloatTy()}),
                               CI->getArgOperand(0), "h2f");
      } else {
        Intrinsic::ID IID = ShouldUpgradeNVPTXBF16Intrinsic(Name);
        if (IID != Intrinsic::not_intrinsic &&
            !F->getReturnType()->getScalarType()->isBFloatTy()) {
          rename(F);
          NewFn = Intrinsic::getDeclaration(F->getParent(), IID);
          SmallVector<Value *, 2> Args;
          for (size_t I = 0; I < NewFn->arg_size(); ++I) {
            Value *Arg = CI->getArgOperand(I);
            Type *OldType = Arg->getType();
            Type *NewType = NewFn->getArg(I)->getType();
            Args.push_back((OldType->isIntegerTy() &&
                            NewType->getScalarType()->isBFloatTy())
                               ? Builder.CreateBitCast(Arg, NewType)
                               : Arg);
          }
          Rep = Builder.CreateCall(NewFn, Args);
          if (F->getReturnType()->isIntegerTy())
            Rep = Builder.CreateBitCast(Rep, F->getReturnType());
        }
      }
    } else if (IsARM) {
      Rep = UpgradeARMIntrinsicCall(Name, CI, F, Builder);
    } else if (IsAMDGCN) {
      Rep = UpgradeAMDGCNIntrinsicCall(Name, CI, F, Builder);
    } else {
      llvm_unreachable("Unknown function for CallBase upgrade.");
    }

    if (Rep)
      CI->replaceAllUsesWith(Rep);
    CI->eraseFromParent();
    return;
  }

  const auto &DefaultCase = [&]() -> void {
    if (CI->getFunctionType() == NewFn->getFunctionType()) {
      // Handle generic mangling change.
      assert(
          (CI->getCalledFunction()->getName() != NewFn->getName()) &&
          "Unknown function for CallBase upgrade and isn't just a name change");
      CI->setCalledFunction(NewFn);
      return;
    }

    // This must be an upgrade from a named to a literal struct.
    if (auto *OldST = dyn_cast<StructType>(CI->getType())) {
      assert(OldST != NewFn->getReturnType() &&
             "Return type must have changed");
      assert(OldST->getNumElements() ==
                 cast<StructType>(NewFn->getReturnType())->getNumElements() &&
             "Must have same number of elements");

      SmallVector<Value *> Args(CI->args());
      Value *NewCI = Builder.CreateCall(NewFn, Args);
      Value *Res = PoisonValue::get(OldST);
      for (unsigned Idx = 0; Idx < OldST->getNumElements(); ++Idx) {
        Value *Elem = Builder.CreateExtractValue(NewCI, Idx);
        Res = Builder.CreateInsertValue(Res, Elem, Idx);
      }
      CI->replaceAllUsesWith(Res);
      CI->eraseFromParent();
      return;
    }

    // We're probably about to produce something invalid. Let the verifier catch
    // it instead of dying here.
    CI->setCalledOperand(
        ConstantExpr::getPointerCast(NewFn, CI->getCalledOperand()->getType()));
    return;
  };
  CallInst *NewCall = nullptr;
  switch (NewFn->getIntrinsicID()) {
  default: {
    DefaultCase();
    return;
  }
  case Intrinsic::arm_neon_vst1:
  case Intrinsic::arm_neon_vst2:
  case Intrinsic::arm_neon_vst3:
  case Intrinsic::arm_neon_vst4:
  case Intrinsic::arm_neon_vst2lane:
  case Intrinsic::arm_neon_vst3lane:
  case Intrinsic::arm_neon_vst4lane: {
    SmallVector<Value *, 4> Args(CI->args());
    NewCall = Builder.CreateCall(NewFn, Args);
    break;
  }
  case Intrinsic::aarch64_sve_bfmlalb_lane_v2:
  case Intrinsic::aarch64_sve_bfmlalt_lane_v2:
  case Intrinsic::aarch64_sve_bfdot_lane_v2: {
    LLVMContext &Ctx = F->getParent()->getContext();
    SmallVector<Value *, 4> Args(CI->args());
    Args[3] = ConstantInt::get(Type::getInt32Ty(Ctx),
                               cast<ConstantInt>(Args[3])->getZExtValue());
    NewCall = Builder.CreateCall(NewFn, Args);
    break;
  }
  case Intrinsic::aarch64_sve_ld3_sret:
  case Intrinsic::aarch64_sve_ld4_sret:
  case Intrinsic::aarch64_sve_ld2_sret: {
    StringRef Name = F->getName();
    Name = Name.substr(5);
    unsigned N = StringSwitch<unsigned>(Name)
                     .StartsWith("aarch64.sve.ld2", 2)
                     .StartsWith("aarch64.sve.ld3", 3)
                     .StartsWith("aarch64.sve.ld4", 4)
                     .Default(0);
    ScalableVectorType *RetTy =
        dyn_cast<ScalableVectorType>(F->getReturnType());
    unsigned MinElts = RetTy->getMinNumElements() / N;
    SmallVector<Value *, 2> Args(CI->args());
    Value *NewLdCall = Builder.CreateCall(NewFn, Args);
    Value *Ret = llvm::PoisonValue::get(RetTy);
    for (unsigned I = 0; I < N; I++) {
      Value *Idx = ConstantInt::get(Type::getInt64Ty(C), I * MinElts);
      Value *SRet = Builder.CreateExtractValue(NewLdCall, I);
      Ret = Builder.CreateInsertVector(RetTy, Ret, SRet, Idx);
    }
    NewCall = dyn_cast<CallInst>(Ret);
    break;
  }

  case Intrinsic::vector_extract: {
    StringRef Name = F->getName();
    Name = Name.substr(5); // Strip llvm
    if (!Name.startswith("aarch64.sve.tuple.get")) {
      DefaultCase();
      return;
    }
    ScalableVectorType *RetTy =
        dyn_cast<ScalableVectorType>(F->getReturnType());
    unsigned MinElts = RetTy->getMinNumElements();
    unsigned I = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
    Value *NewIdx = ConstantInt::get(Type::getInt64Ty(C), I * MinElts);
    NewCall = Builder.CreateCall(NewFn, {CI->getArgOperand(0), NewIdx});
    break;
  }

  case Intrinsic::vector_insert: {
    StringRef Name = F->getName();
    Name = Name.substr(5);
    if (!Name.startswith("aarch64.sve.tuple")) {
      DefaultCase();
      return;
    }
    if (Name.startswith("aarch64.sve.tuple.set")) {
      unsigned I = dyn_cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      ScalableVectorType *Ty =
          dyn_cast<ScalableVectorType>(CI->getArgOperand(2)->getType());
      Value *NewIdx =
          ConstantInt::get(Type::getInt64Ty(C), I * Ty->getMinNumElements());
      NewCall = Builder.CreateCall(
          NewFn, {CI->getArgOperand(0), CI->getArgOperand(2), NewIdx});
      break;
    }
    if (Name.startswith("aarch64.sve.tuple.create")) {
      unsigned N = StringSwitch<unsigned>(Name)
                       .StartsWith("aarch64.sve.tuple.create2", 2)
                       .StartsWith("aarch64.sve.tuple.create3", 3)
                       .StartsWith("aarch64.sve.tuple.create4", 4)
                       .Default(0);
      assert(N > 1 && "Create is expected to be between 2-4");
      ScalableVectorType *RetTy =
          dyn_cast<ScalableVectorType>(F->getReturnType());
      Value *Ret = llvm::PoisonValue::get(RetTy);
      unsigned MinElts = RetTy->getMinNumElements() / N;
      for (unsigned I = 0; I < N; I++) {
        Value *Idx = ConstantInt::get(Type::getInt64Ty(C), I * MinElts);
        Value *V = CI->getArgOperand(I);
        Ret = Builder.CreateInsertVector(RetTy, Ret, V, Idx);
      }
      NewCall = dyn_cast<CallInst>(Ret);
    }
    break;
  }

  case Intrinsic::arm_neon_bfdot:
  case Intrinsic::arm_neon_bfmmla:
  case Intrinsic::arm_neon_bfmlalb:
  case Intrinsic::arm_neon_bfmlalt:
  case Intrinsic::aarch64_neon_bfdot:
  case Intrinsic::aarch64_neon_bfmmla:
  case Intrinsic::aarch64_neon_bfmlalb:
  case Intrinsic::aarch64_neon_bfmlalt: {
    SmallVector<Value *, 3> Args;
    assert(CI->arg_size() == 3 &&
           "Mismatch between function args and call args");
    size_t OperandWidth =
        CI->getArgOperand(1)->getType()->getPrimitiveSizeInBits();
    assert((OperandWidth == 64 || OperandWidth == 128) &&
           "Unexpected operand width");
    Type *NewTy = FixedVectorType::get(Type::getBFloatTy(C), OperandWidth / 16);
    auto Iter = CI->args().begin();
    Args.push_back(*Iter++);
    Args.push_back(Builder.CreateBitCast(*Iter++, NewTy));
    Args.push_back(Builder.CreateBitCast(*Iter++, NewTy));
    NewCall = Builder.CreateCall(NewFn, Args);
    break;
  }

  case Intrinsic::bitreverse:
    NewCall = Builder.CreateCall(NewFn, {CI->getArgOperand(0)});
    break;

  case Intrinsic::ctlz:
  case Intrinsic::cttz:
    assert(CI->arg_size() == 1 &&
           "Mismatch between function args and call args");
    NewCall =
        Builder.CreateCall(NewFn, {CI->getArgOperand(0), Builder.getFalse()});
    break;

  case Intrinsic::objectsize: {
    Value *NullIsUnknownSize =
        CI->arg_size() == 2 ? Builder.getFalse() : CI->getArgOperand(2);
    Value *Dynamic =
        CI->arg_size() < 4 ? Builder.getFalse() : CI->getArgOperand(3);
    NewCall = Builder.CreateCall(
        NewFn, {CI->getArgOperand(0), CI->getArgOperand(1), NullIsUnknownSize, Dynamic});
    break;
  }

  case Intrinsic::ctpop:
    NewCall = Builder.CreateCall(NewFn, {CI->getArgOperand(0)});
    break;

  case Intrinsic::convert_from_fp16:
    NewCall = Builder.CreateCall(NewFn, {CI->getArgOperand(0)});
    break;

  case Intrinsic::dbg_value: {
    StringRef Name = F->getName();
    Name = Name.substr(5); // Strip llvm.
    // Upgrade `dbg.addr` to `dbg.value` with `DW_OP_deref`.
    if (Name.startswith("dbg.addr")) {
      DIExpression *Expr = cast<DIExpression>(
          cast<MetadataAsValue>(CI->getArgOperand(2))->getMetadata());
      Expr = DIExpression::append(Expr, dwarf::DW_OP_deref);
      NewCall =
          Builder.CreateCall(NewFn, {CI->getArgOperand(0), CI->getArgOperand(1),
                                     MetadataAsValue::get(C, Expr)});
      break;
    }

    // Upgrade from the old version that had an extra offset argument.
    assert(CI->arg_size() == 4);
    // Drop nonzero offsets instead of attempting to upgrade them.
    if (auto *Offset = dyn_cast_or_null<Constant>(CI->getArgOperand(1)))
      if (Offset->isZeroValue()) {
        NewCall = Builder.CreateCall(
            NewFn,
            {CI->getArgOperand(0), CI->getArgOperand(2), CI->getArgOperand(3)});
        break;
      }
    CI->eraseFromParent();
    return;
  }

  case Intrinsic::ptr_annotation:
    // Upgrade from versions that lacked the annotation attribute argument.
    if (CI->arg_size() != 4) {
      DefaultCase();
      return;
    }

    // Create a new call with an added null annotation attribute argument.
    NewCall = Builder.CreateCall(
        NewFn,
        {CI->getArgOperand(0), CI->getArgOperand(1), CI->getArgOperand(2),
         CI->getArgOperand(3), Constant::getNullValue(Builder.getInt8PtrTy())});
    NewCall->takeName(CI);
    CI->replaceAllUsesWith(NewCall);
    CI->eraseFromParent();
    return;

  case Intrinsic::var_annotation:
    // Upgrade from versions that lacked the annotation attribute argument.
    if (CI->arg_size() != 4) {
      DefaultCase();
      return;
    }
    // Create a new call with an added null annotation attribute argument.
    NewCall = Builder.CreateCall(
        NewFn,
        {CI->getArgOperand(0), CI->getArgOperand(1), CI->getArgOperand(2),
         CI->getArgOperand(3), Constant::getNullValue(Builder.getInt8PtrTy())});
    NewCall->takeName(CI);
    CI->replaceAllUsesWith(NewCall);
    CI->eraseFromParent();
    return;

  case Intrinsic::riscv_aes32dsi:
  case Intrinsic::riscv_aes32dsmi:
  case Intrinsic::riscv_aes32esi:
  case Intrinsic::riscv_aes32esmi:
  case Intrinsic::riscv_sm4ks:
  case Intrinsic::riscv_sm4ed: {
    // The last argument to these intrinsics used to be i8 and changed to i32.
    // The type overload for sm4ks and sm4ed was removed.
    Value *Arg2 = CI->getArgOperand(2);
    if (Arg2->getType()->isIntegerTy(32) && !CI->getType()->isIntegerTy(64))
      return;

    Value *Arg0 = CI->getArgOperand(0);
    Value *Arg1 = CI->getArgOperand(1);
    if (CI->getType()->isIntegerTy(64)) {
      Arg0 = Builder.CreateTrunc(Arg0, Builder.getInt32Ty());
      Arg1 = Builder.CreateTrunc(Arg1, Builder.getInt32Ty());
    }

    Arg2 = ConstantInt::get(Type::getInt32Ty(C),
                            cast<ConstantInt>(Arg2)->getZExtValue());

    NewCall = Builder.CreateCall(NewFn, {Arg0, Arg1, Arg2});
    Value *Res = NewCall;
    if (Res->getType() != CI->getType())
      Res = Builder.CreateIntCast(NewCall, CI->getType(), /*isSigned*/ true);
    NewCall->takeName(CI);
    CI->replaceAllUsesWith(Res);
    CI->eraseFromParent();
    return;
  }
  case Intrinsic::riscv_sha256sig0:
  case Intrinsic::riscv_sha256sig1:
  case Intrinsic::riscv_sha256sum0:
  case Intrinsic::riscv_sha256sum1:
  case Intrinsic::riscv_sm3p0:
  case Intrinsic::riscv_sm3p1: {
    // The last argument to these intrinsics used to be i8 and changed to i32.
    // The type overload for sm4ks and sm4ed was removed.
    if (!CI->getType()->isIntegerTy(64))
      return;

    Value *Arg =
        Builder.CreateTrunc(CI->getArgOperand(0), Builder.getInt32Ty());

    NewCall = Builder.CreateCall(NewFn, Arg);
    Value *Res =
        Builder.CreateIntCast(NewCall, CI->getType(), /*isSigned*/ true);
    NewCall->takeName(CI);
    CI->replaceAllUsesWith(Res);
    CI->eraseFromParent();
    return;
  }

  case Intrinsic::x86_xop_vfrcz_ss:
  case Intrinsic::x86_xop_vfrcz_sd:
    NewCall = Builder.CreateCall(NewFn, {CI->getArgOperand(1)});
    break;

  case Intrinsic::x86_xop_vpermil2pd:
  case Intrinsic::x86_xop_vpermil2ps:
  case Intrinsic::x86_xop_vpermil2pd_256:
  case Intrinsic::x86_xop_vpermil2ps_256: {
    SmallVector<Value *, 4> Args(CI->args());
    VectorType *FltIdxTy = cast<VectorType>(Args[2]->getType());
    VectorType *IntIdxTy = VectorType::getInteger(FltIdxTy);
    Args[2] = Builder.CreateBitCast(Args[2], IntIdxTy);
    NewCall = Builder.CreateCall(NewFn, Args);
    break;
  }

  case Intrinsic::x86_sse41_ptestc:
  case Intrinsic::x86_sse41_ptestz:
  case Intrinsic::x86_sse41_ptestnzc: {
    // The arguments for these intrinsics used to be v4f32, and changed
    // to v2i64. This is purely a nop, since those are bitwise intrinsics.
    // So, the only thing required is a bitcast for both arguments.
    // First, check the arguments have the old type.
    Value *Arg0 = CI->getArgOperand(0);
    if (Arg0->getType() != FixedVectorType::get(Type::getFloatTy(C), 4))
      return;

    // Old intrinsic, add bitcasts
    Value *Arg1 = CI->getArgOperand(1);

    auto *NewVecTy = FixedVectorType::get(Type::getInt64Ty(C), 2);

    Value *BC0 = Builder.CreateBitCast(Arg0, NewVecTy, "cast");
    Value *BC1 = Builder.CreateBitCast(Arg1, NewVecTy, "cast");

    NewCall = Builder.CreateCall(NewFn, {BC0, BC1});
    break;
  }

  case Intrinsic::x86_rdtscp: {
    // This used to take 1 arguments. If we have no arguments, it is already
    // upgraded.
    if (CI->getNumOperands() == 0)
      return;

    NewCall = Builder.CreateCall(NewFn);
    // Extract the second result and store it.
    Value *Data = Builder.CreateExtractValue(NewCall, 1);
    // Cast the pointer to the right type.
    Value *Ptr = Builder.CreateBitCast(CI->getArgOperand(0),
                                 llvm::PointerType::getUnqual(Data->getType()));
    Builder.CreateAlignedStore(Data, Ptr, Align(1));
    // Replace the original call result with the first result of the new call.
    Value *TSC = Builder.CreateExtractValue(NewCall, 0);

    NewCall->takeName(CI);
    CI->replaceAllUsesWith(TSC);
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
    SmallVector<Value *, 4> Args(CI->args());

    // Replace the last argument with a trunc.
    Args.back() = Builder.CreateTrunc(Args.back(), Type::getInt8Ty(C), "trunc");
    NewCall = Builder.CreateCall(NewFn, Args);
    break;
  }

  case Intrinsic::x86_avx512_mask_cmp_pd_128:
  case Intrinsic::x86_avx512_mask_cmp_pd_256:
  case Intrinsic::x86_avx512_mask_cmp_pd_512:
  case Intrinsic::x86_avx512_mask_cmp_ps_128:
  case Intrinsic::x86_avx512_mask_cmp_ps_256:
  case Intrinsic::x86_avx512_mask_cmp_ps_512: {
    SmallVector<Value *, 4> Args(CI->args());
    unsigned NumElts =
        cast<FixedVectorType>(Args[0]->getType())->getNumElements();
    Args[3] = getX86MaskVec(Builder, Args[3], NumElts);

    NewCall = Builder.CreateCall(NewFn, Args);
    Value *Res = ApplyX86MaskOn1BitsVec(Builder, NewCall, nullptr);

    NewCall->takeName(CI);
    CI->replaceAllUsesWith(Res);
    CI->eraseFromParent();
    return;
  }

  case Intrinsic::x86_avx512bf16_cvtne2ps2bf16_128:
  case Intrinsic::x86_avx512bf16_cvtne2ps2bf16_256:
  case Intrinsic::x86_avx512bf16_cvtne2ps2bf16_512:
  case Intrinsic::x86_avx512bf16_mask_cvtneps2bf16_128:
  case Intrinsic::x86_avx512bf16_cvtneps2bf16_256:
  case Intrinsic::x86_avx512bf16_cvtneps2bf16_512: {
    SmallVector<Value *, 4> Args(CI->args());
    unsigned NumElts = cast<FixedVectorType>(CI->getType())->getNumElements();
    if (NewFn->getIntrinsicID() ==
        Intrinsic::x86_avx512bf16_mask_cvtneps2bf16_128)
      Args[1] = Builder.CreateBitCast(
          Args[1], FixedVectorType::get(Builder.getBFloatTy(), NumElts));

    NewCall = Builder.CreateCall(NewFn, Args);
    Value *Res = Builder.CreateBitCast(
        NewCall, FixedVectorType::get(Builder.getInt16Ty(), NumElts));

    NewCall->takeName(CI);
    CI->replaceAllUsesWith(Res);
    CI->eraseFromParent();
    return;
  }
  case Intrinsic::x86_avx512bf16_dpbf16ps_128:
  case Intrinsic::x86_avx512bf16_dpbf16ps_256:
  case Intrinsic::x86_avx512bf16_dpbf16ps_512:{
    SmallVector<Value *, 4> Args(CI->args());
    unsigned NumElts =
        cast<FixedVectorType>(CI->getType())->getNumElements() * 2;
    Args[1] = Builder.CreateBitCast(
        Args[1], FixedVectorType::get(Builder.getBFloatTy(), NumElts));
    Args[2] = Builder.CreateBitCast(
        Args[2], FixedVectorType::get(Builder.getBFloatTy(), NumElts));

    NewCall = Builder.CreateCall(NewFn, Args);
    break;
  }

  case Intrinsic::thread_pointer: {
    NewCall = Builder.CreateCall(NewFn, {});
    break;
  }

  case Intrinsic::invariant_start:
  case Intrinsic::invariant_end: {
    SmallVector<Value *, 4> Args(CI->args());
    NewCall = Builder.CreateCall(NewFn, Args);
    break;
  }
  case Intrinsic::masked_load:
  case Intrinsic::masked_store:
  case Intrinsic::masked_gather:
  case Intrinsic::masked_scatter: {
    SmallVector<Value *, 4> Args(CI->args());
    NewCall = Builder.CreateCall(NewFn, Args);
    NewCall->copyMetadata(*CI);
    break;
  }

  case Intrinsic::memcpy:
  case Intrinsic::memmove:
  case Intrinsic::memset: {
    // We have to make sure that the call signature is what we're expecting.
    // We only want to change the old signatures by removing the alignment arg:
    //  @llvm.mem[cpy|move]...(i8*, i8*, i[32|i64], i32, i1)
    //    -> @llvm.mem[cpy|move]...(i8*, i8*, i[32|i64], i1)
    //  @llvm.memset...(i8*, i8, i[32|64], i32, i1)
    //    -> @llvm.memset...(i8*, i8, i[32|64], i1)
    // Note: i8*'s in the above can be any pointer type
    if (CI->arg_size() != 5) {
      DefaultCase();
      return;
    }
    // Remove alignment argument (3), and add alignment attributes to the
    // dest/src pointers.
    Value *Args[4] = {CI->getArgOperand(0), CI->getArgOperand(1),
                      CI->getArgOperand(2), CI->getArgOperand(4)};
    NewCall = Builder.CreateCall(NewFn, Args);
    AttributeList OldAttrs = CI->getAttributes();
    AttributeList NewAttrs = AttributeList::get(
        C, OldAttrs.getFnAttrs(), OldAttrs.getRetAttrs(),
        {OldAttrs.getParamAttrs(0), OldAttrs.getParamAttrs(1),
         OldAttrs.getParamAttrs(2), OldAttrs.getParamAttrs(4)});
    NewCall->setAttributes(NewAttrs);
    auto *MemCI = cast<MemIntrinsic>(NewCall);
    // All mem intrinsics support dest alignment.
    const ConstantInt *Align = cast<ConstantInt>(CI->getArgOperand(3));
    MemCI->setDestAlignment(Align->getMaybeAlignValue());
    // Memcpy/Memmove also support source alignment.
    if (auto *MTI = dyn_cast<MemTransferInst>(MemCI))
      MTI->setSourceAlignment(Align->getMaybeAlignValue());
    break;
  }
  }
  assert(NewCall && "Should have either set this variable or returned through "
                    "the default case");
  NewCall->takeName(CI);
  CI->replaceAllUsesWith(NewCall);
  CI->eraseFromParent();
}

void llvm::UpgradeCallsToIntrinsic(Function *F) {
  assert(F && "Illegal attempt to upgrade a non-existent intrinsic.");

  // Check if this function should be upgraded and get the replacement function
  // if there is one.
  Function *NewFn;
  if (UpgradeIntrinsicFunction(F, NewFn)) {
    // Replace all users of the old function with the new function or new
    // instructions. This is not a range loop because the call is deleted.
    for (User *U : make_early_inc_range(F->users()))
      if (CallBase *CB = dyn_cast<CallBase>(U))
        UpgradeIntrinsicCall(CB, NewFn);

    // Remove old function, no longer used, from the module.
    F->eraseFromParent();
  }
}

MDNode *llvm::UpgradeTBAANode(MDNode &MD) {
  const unsigned NumOperands = MD.getNumOperands();
  if (NumOperands == 0)
    return &MD; // Invalid, punt to a verifier error.

  // Check if the tag uses struct-path aware TBAA format.
  if (isa<MDNode>(MD.getOperand(0)) && NumOperands >= 3)
    return &MD;

  auto &Context = MD.getContext();
  if (NumOperands == 3) {
    Metadata *Elts[] = {MD.getOperand(0), MD.getOperand(1)};
    MDNode *ScalarType = MDNode::get(Context, Elts);
    // Create a MDNode <ScalarType, ScalarType, offset 0, const>
    Metadata *Elts2[] = {ScalarType, ScalarType,
                         ConstantAsMetadata::get(
                             Constant::getNullValue(Type::getInt64Ty(Context))),
                         MD.getOperand(2)};
    return MDNode::get(Context, Elts2);
  }
  // Create a MDNode <MD, MD, offset 0>
  Metadata *Elts[] = {&MD, &MD, ConstantAsMetadata::get(Constant::getNullValue(
                                    Type::getInt64Ty(Context)))};
  return MDNode::get(Context, Elts);
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

Constant *llvm::UpgradeBitCastExpr(unsigned Opc, Constant *C, Type *DestTy) {
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
  if (DisableAutoUpgradeDebugInfo)
    return false;

  unsigned Version = getDebugMetadataVersionFromModule(M);
  if (Version == DEBUG_METADATA_VERSION) {
    bool BrokenDebugInfo = false;
    if (verifyModule(M, &llvm::errs(), &BrokenDebugInfo))
      report_fatal_error("Broken module found, compilation aborted!");
    if (!BrokenDebugInfo)
      // Everything is ok.
      return false;
    else {
      // Diagnose malformed debug info.
      DiagnosticInfoIgnoringInvalidDebugMetadata Diag(M);
      M.getContext().diagnose(Diag);
    }
  }
  bool Modified = StripDebugInfo(M);
  if (Modified && Version != DEBUG_METADATA_VERSION) {
    // Diagnose a version mismatch.
    DiagnosticInfoDebugMetadataVersion DiagVersion(M, Version);
    M.getContext().diagnose(DiagVersion);
  }
  return Modified;
}

/// This checks for objc retain release marker which should be upgraded. It
/// returns true if module is modified.
static bool UpgradeRetainReleaseMarker(Module &M) {
  bool Changed = false;
  const char *MarkerKey = "clang.arc.retainAutoreleasedReturnValueMarker";
  NamedMDNode *ModRetainReleaseMarker = M.getNamedMetadata(MarkerKey);
  if (ModRetainReleaseMarker) {
    MDNode *Op = ModRetainReleaseMarker->getOperand(0);
    if (Op) {
      MDString *ID = dyn_cast_or_null<MDString>(Op->getOperand(0));
      if (ID) {
        SmallVector<StringRef, 4> ValueComp;
        ID->getString().split(ValueComp, "#");
        if (ValueComp.size() == 2) {
          std::string NewValue = ValueComp[0].str() + ";" + ValueComp[1].str();
          ID = MDString::get(M.getContext(), NewValue);
        }
        M.addModuleFlag(Module::Error, MarkerKey, ID);
        M.eraseNamedMetadata(ModRetainReleaseMarker);
        Changed = true;
      }
    }
  }
  return Changed;
}

void llvm::UpgradeARCRuntime(Module &M) {
  // This lambda converts normal function calls to ARC runtime functions to
  // intrinsic calls.
  auto UpgradeToIntrinsic = [&](const char *OldFunc,
                                llvm::Intrinsic::ID IntrinsicFunc) {
    Function *Fn = M.getFunction(OldFunc);

    if (!Fn)
      return;

    Function *NewFn = llvm::Intrinsic::getDeclaration(&M, IntrinsicFunc);

    for (User *U : make_early_inc_range(Fn->users())) {
      CallInst *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->getCalledFunction() != Fn)
        continue;

      IRBuilder<> Builder(CI->getParent(), CI->getIterator());
      FunctionType *NewFuncTy = NewFn->getFunctionType();
      SmallVector<Value *, 2> Args;

      // Don't upgrade the intrinsic if it's not valid to bitcast the return
      // value to the return type of the old function.
      if (NewFuncTy->getReturnType() != CI->getType() &&
          !CastInst::castIsValid(Instruction::BitCast, CI,
                                 NewFuncTy->getReturnType()))
        continue;

      bool InvalidCast = false;

      for (unsigned I = 0, E = CI->arg_size(); I != E; ++I) {
        Value *Arg = CI->getArgOperand(I);

        // Bitcast argument to the parameter type of the new function if it's
        // not a variadic argument.
        if (I < NewFuncTy->getNumParams()) {
          // Don't upgrade the intrinsic if it's not valid to bitcast the argument
          // to the parameter type of the new function.
          if (!CastInst::castIsValid(Instruction::BitCast, Arg,
                                     NewFuncTy->getParamType(I))) {
            InvalidCast = true;
            break;
          }
          Arg = Builder.CreateBitCast(Arg, NewFuncTy->getParamType(I));
        }
        Args.push_back(Arg);
      }

      if (InvalidCast)
        continue;

      // Create a call instruction that calls the new function.
      CallInst *NewCall = Builder.CreateCall(NewFuncTy, NewFn, Args);
      NewCall->setTailCallKind(cast<CallInst>(CI)->getTailCallKind());
      NewCall->takeName(CI);

      // Bitcast the return value back to the type of the old call.
      Value *NewRetVal = Builder.CreateBitCast(NewCall, CI->getType());

      if (!CI->use_empty())
        CI->replaceAllUsesWith(NewRetVal);
      CI->eraseFromParent();
    }

    if (Fn->use_empty())
      Fn->eraseFromParent();
  };

  // Unconditionally convert a call to "clang.arc.use" to a call to
  // "llvm.objc.clang.arc.use".
  UpgradeToIntrinsic("clang.arc.use", llvm::Intrinsic::objc_clang_arc_use);

  // Upgrade the retain release marker. If there is no need to upgrade
  // the marker, that means either the module is already new enough to contain
  // new intrinsics or it is not ARC. There is no need to upgrade runtime call.
  if (!UpgradeRetainReleaseMarker(M))
    return;

  std::pair<const char *, llvm::Intrinsic::ID> RuntimeFuncs[] = {
      {"objc_autorelease", llvm::Intrinsic::objc_autorelease},
      {"objc_autoreleasePoolPop", llvm::Intrinsic::objc_autoreleasePoolPop},
      {"objc_autoreleasePoolPush", llvm::Intrinsic::objc_autoreleasePoolPush},
      {"objc_autoreleaseReturnValue",
       llvm::Intrinsic::objc_autoreleaseReturnValue},
      {"objc_copyWeak", llvm::Intrinsic::objc_copyWeak},
      {"objc_destroyWeak", llvm::Intrinsic::objc_destroyWeak},
      {"objc_initWeak", llvm::Intrinsic::objc_initWeak},
      {"objc_loadWeak", llvm::Intrinsic::objc_loadWeak},
      {"objc_loadWeakRetained", llvm::Intrinsic::objc_loadWeakRetained},
      {"objc_moveWeak", llvm::Intrinsic::objc_moveWeak},
      {"objc_release", llvm::Intrinsic::objc_release},
      {"objc_retain", llvm::Intrinsic::objc_retain},
      {"objc_retainAutorelease", llvm::Intrinsic::objc_retainAutorelease},
      {"objc_retainAutoreleaseReturnValue",
       llvm::Intrinsic::objc_retainAutoreleaseReturnValue},
      {"objc_retainAutoreleasedReturnValue",
       llvm::Intrinsic::objc_retainAutoreleasedReturnValue},
      {"objc_retainBlock", llvm::Intrinsic::objc_retainBlock},
      {"objc_storeStrong", llvm::Intrinsic::objc_storeStrong},
      {"objc_storeWeak", llvm::Intrinsic::objc_storeWeak},
      {"objc_unsafeClaimAutoreleasedReturnValue",
       llvm::Intrinsic::objc_unsafeClaimAutoreleasedReturnValue},
      {"objc_retainedObject", llvm::Intrinsic::objc_retainedObject},
      {"objc_unretainedObject", llvm::Intrinsic::objc_unretainedObject},
      {"objc_unretainedPointer", llvm::Intrinsic::objc_unretainedPointer},
      {"objc_retain_autorelease", llvm::Intrinsic::objc_retain_autorelease},
      {"objc_sync_enter", llvm::Intrinsic::objc_sync_enter},
      {"objc_sync_exit", llvm::Intrinsic::objc_sync_exit},
      {"objc_arc_annotation_topdown_bbstart",
       llvm::Intrinsic::objc_arc_annotation_topdown_bbstart},
      {"objc_arc_annotation_topdown_bbend",
       llvm::Intrinsic::objc_arc_annotation_topdown_bbend},
      {"objc_arc_annotation_bottomup_bbstart",
       llvm::Intrinsic::objc_arc_annotation_bottomup_bbstart},
      {"objc_arc_annotation_bottomup_bbend",
       llvm::Intrinsic::objc_arc_annotation_bottomup_bbend}};

  for (auto &I : RuntimeFuncs)
    UpgradeToIntrinsic(I.first, I.second);
}

bool llvm::UpgradeModuleFlags(Module &M) {
  NamedMDNode *ModFlags = M.getModuleFlagsMetadata();
  if (!ModFlags)
    return false;

  bool HasObjCFlag = false, HasClassProperties = false, Changed = false;
  bool HasSwiftVersionFlag = false;
  uint8_t SwiftMajorVersion, SwiftMinorVersion;
  uint32_t SwiftABIVersion;
  auto Int8Ty = Type::getInt8Ty(M.getContext());
  auto Int32Ty = Type::getInt32Ty(M.getContext());

  for (unsigned I = 0, E = ModFlags->getNumOperands(); I != E; ++I) {
    MDNode *Op = ModFlags->getOperand(I);
    if (Op->getNumOperands() != 3)
      continue;
    MDString *ID = dyn_cast_or_null<MDString>(Op->getOperand(1));
    if (!ID)
      continue;
    auto SetBehavior = [&](Module::ModFlagBehavior B) {
      Metadata *Ops[3] = {ConstantAsMetadata::get(ConstantInt::get(
                              Type::getInt32Ty(M.getContext()), B)),
                          MDString::get(M.getContext(), ID->getString()),
                          Op->getOperand(2)};
      ModFlags->setOperand(I, MDNode::get(M.getContext(), Ops));
      Changed = true;
    };

    if (ID->getString() == "Objective-C Image Info Version")
      HasObjCFlag = true;
    if (ID->getString() == "Objective-C Class Properties")
      HasClassProperties = true;
    // Upgrade PIC from Error/Max to Min.
    if (ID->getString() == "PIC Level") {
      if (auto *Behavior =
              mdconst::dyn_extract_or_null<ConstantInt>(Op->getOperand(0))) {
        uint64_t V = Behavior->getLimitedValue();
        if (V == Module::Error || V == Module::Max)
          SetBehavior(Module::Min);
      }
    }
    // Upgrade "PIE Level" from Error to Max.
    if (ID->getString() == "PIE Level")
      if (auto *Behavior =
              mdconst::dyn_extract_or_null<ConstantInt>(Op->getOperand(0)))
        if (Behavior->getLimitedValue() == Module::Error)
          SetBehavior(Module::Max);

    // Upgrade branch protection and return address signing module flags. The
    // module flag behavior for these fields were Error and now they are Min.
    if (ID->getString() == "branch-target-enforcement" ||
        ID->getString().startswith("sign-return-address")) {
      if (auto *Behavior =
              mdconst::dyn_extract_or_null<ConstantInt>(Op->getOperand(0))) {
        if (Behavior->getLimitedValue() == Module::Error) {
          Type *Int32Ty = Type::getInt32Ty(M.getContext());
          Metadata *Ops[3] = {
              ConstantAsMetadata::get(ConstantInt::get(Int32Ty, Module::Min)),
              Op->getOperand(1), Op->getOperand(2)};
          ModFlags->setOperand(I, MDNode::get(M.getContext(), Ops));
          Changed = true;
        }
      }
    }

    // Upgrade Objective-C Image Info Section. Removed the whitespce in the
    // section name so that llvm-lto will not complain about mismatching
    // module flags that is functionally the same.
    if (ID->getString() == "Objective-C Image Info Section") {
      if (auto *Value = dyn_cast_or_null<MDString>(Op->getOperand(2))) {
        SmallVector<StringRef, 4> ValueComp;
        Value->getString().split(ValueComp, " ");
        if (ValueComp.size() != 1) {
          std::string NewValue;
          for (auto &S : ValueComp)
            NewValue += S.str();
          Metadata *Ops[3] = {Op->getOperand(0), Op->getOperand(1),
                              MDString::get(M.getContext(), NewValue)};
          ModFlags->setOperand(I, MDNode::get(M.getContext(), Ops));
          Changed = true;
        }
      }
    }

    // IRUpgrader turns a i32 type "Objective-C Garbage Collection" into i8 value.
    // If the higher bits are set, it adds new module flag for swift info.
    if (ID->getString() == "Objective-C Garbage Collection") {
      auto Md = dyn_cast<ConstantAsMetadata>(Op->getOperand(2));
      if (Md) {
        assert(Md->getValue() && "Expected non-empty metadata");
        auto Type = Md->getValue()->getType();
        if (Type == Int8Ty)
          continue;
        unsigned Val = Md->getValue()->getUniqueInteger().getZExtValue();
        if ((Val & 0xff) != Val) {
          HasSwiftVersionFlag = true;
          SwiftABIVersion = (Val & 0xff00) >> 8;
          SwiftMajorVersion = (Val & 0xff000000) >> 24;
          SwiftMinorVersion = (Val & 0xff0000) >> 16;
        }
        Metadata *Ops[3] = {
          ConstantAsMetadata::get(ConstantInt::get(Int32Ty,Module::Error)),
          Op->getOperand(1),
          ConstantAsMetadata::get(ConstantInt::get(Int8Ty,Val & 0xff))};
        ModFlags->setOperand(I, MDNode::get(M.getContext(), Ops));
        Changed = true;
      }
    }
  }

  // "Objective-C Class Properties" is recently added for Objective-C. We
  // upgrade ObjC bitcodes to contain a "Objective-C Class Properties" module
  // flag of value 0, so we can correclty downgrade this flag when trying to
  // link an ObjC bitcode without this module flag with an ObjC bitcode with
  // this module flag.
  if (HasObjCFlag && !HasClassProperties) {
    M.addModuleFlag(llvm::Module::Override, "Objective-C Class Properties",
                    (uint32_t)0);
    Changed = true;
  }

  if (HasSwiftVersionFlag) {
    M.addModuleFlag(Module::Error, "Swift ABI Version",
                    SwiftABIVersion);
    M.addModuleFlag(Module::Error, "Swift Major Version",
                    ConstantInt::get(Int8Ty, SwiftMajorVersion));
    M.addModuleFlag(Module::Error, "Swift Minor Version",
                    ConstantInt::get(Int8Ty, SwiftMinorVersion));
    Changed = true;
  }

  return Changed;
}

void llvm::UpgradeSectionAttributes(Module &M) {
  auto TrimSpaces = [](StringRef Section) -> std::string {
    SmallVector<StringRef, 5> Components;
    Section.split(Components, ',');

    SmallString<32> Buffer;
    raw_svector_ostream OS(Buffer);

    for (auto Component : Components)
      OS << ',' << Component.trim();

    return std::string(OS.str().substr(1));
  };

  for (auto &GV : M.globals()) {
    if (!GV.hasSection())
      continue;

    StringRef Section = GV.getSection();

    if (!Section.startswith("__DATA, __objc_catlist"))
      continue;

    // __DATA, __objc_catlist, regular, no_dead_strip
    // __DATA,__objc_catlist,regular,no_dead_strip
    GV.setSection(TrimSpaces(Section));
  }
}

namespace {
// Prior to LLVM 10.0, the strictfp attribute could be used on individual
// callsites within a function that did not also have the strictfp attribute.
// Since 10.0, if strict FP semantics are needed within a function, the
// function must have the strictfp attribute and all calls within the function
// must also have the strictfp attribute. This latter restriction is
// necessary to prevent unwanted libcall simplification when a function is
// being cloned (such as for inlining).
//
// The "dangling" strictfp attribute usage was only used to prevent constant
// folding and other libcall simplification. The nobuiltin attribute on the
// callsite has the same effect.
struct StrictFPUpgradeVisitor : public InstVisitor<StrictFPUpgradeVisitor> {
  StrictFPUpgradeVisitor() = default;

  void visitCallBase(CallBase &Call) {
    if (!Call.isStrictFP())
      return;
    if (isa<ConstrainedFPIntrinsic>(&Call))
      return;
    // If we get here, the caller doesn't have the strictfp attribute
    // but this callsite does. Replace the strictfp attribute with nobuiltin.
    Call.removeFnAttr(Attribute::StrictFP);
    Call.addFnAttr(Attribute::NoBuiltin);
  }
};
} // namespace

void llvm::UpgradeFunctionAttributes(Function &F) {
  // If a function definition doesn't have the strictfp attribute,
  // convert any callsite strictfp attributes to nobuiltin.
  if (!F.isDeclaration() && !F.hasFnAttribute(Attribute::StrictFP)) {
    StrictFPUpgradeVisitor SFPV;
    SFPV.visit(F);
  }

  // Remove all incompatibile attributes from function.
  F.removeRetAttrs(AttributeFuncs::typeIncompatible(F.getReturnType()));
  for (auto &Arg : F.args())
    Arg.removeAttrs(AttributeFuncs::typeIncompatible(Arg.getType()));
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

  if (none_of(T->operands(), isOldLoopArgument))
    return &N;

  SmallVector<Metadata *, 8> Ops;
  Ops.reserve(T->getNumOperands());
  for (Metadata *MD : T->operands())
    Ops.push_back(upgradeLoopArgument(MD));

  return MDTuple::get(T->getContext(), Ops);
}

std::string llvm::UpgradeDataLayoutString(StringRef DL, StringRef TT) {
  Triple T(TT);
  // The only data layout upgrades needed for pre-GCN are setting the address
  // space of globals to 1.
  if (T.isAMDGPU() && !T.isAMDGCN() && !DL.contains("-G") &&
      !DL.startswith("G")) {
    return DL.empty() ? std::string("G1") : (DL + "-G1").str();
  }

  if (T.isRISCV64()) {
    // Make i32 a native type for 64-bit RISC-V.
    auto I = DL.find("-n64-");
    if (I != StringRef::npos)
      return (DL.take_front(I) + "-n32:64-" + DL.drop_front(I + 5)).str();
    return DL.str();
  }

  std::string Res = DL.str();
  // AMDGCN data layout upgrades.
  if (T.isAMDGCN()) {
    // Define address spaces for constants.
    if (!DL.contains("-G") && !DL.starts_with("G"))
      Res.append(Res.empty() ? "G1" : "-G1");

    // Add missing non-integral declarations.
    // This goes before adding new address spaces to prevent incoherent string
    // values.
    if (!DL.contains("-ni") && !DL.startswith("ni"))
      Res.append("-ni:7:8");
    // Update ni:7 to ni:7:8.
    if (DL.ends_with("ni:7"))
      Res.append(":8");

    // Add sizing for address spaces 7 and 8 (fat raw buffers and buffer
    // resources) An empty data layout has already been upgraded to G1 by now.
    if (!DL.contains("-p7") && !DL.startswith("p7"))
      Res.append("-p7:160:256:256:32");
    if (!DL.contains("-p8") && !DL.startswith("p8"))
      Res.append("-p8:128:128");

    return Res;
  }

  if (!T.isX86())
    return Res;

  // If the datalayout matches the expected format, add pointer size address
  // spaces to the datalayout.
  std::string AddrSpaces = "-p270:32:32-p271:32:32-p272:64:64";
  if (!DL.contains(AddrSpaces)) {
    SmallVector<StringRef, 4> Groups;
    Regex R("(e-m:[a-z](-p:32:32)?)(-[if]64:.*$)");
    if (R.match(DL, &Groups))
      Res = (Groups[1] + AddrSpaces + Groups[3]).str();
  }

  // For 32-bit MSVC targets, raise the alignment of f80 values to 16 bytes.
  // Raising the alignment is safe because Clang did not produce f80 values in
  // the MSVC environment before this upgrade was added.
  if (T.isWindowsMSVCEnvironment() && !T.isArch64Bit()) {
    StringRef Ref = Res;
    auto I = Ref.find("-f80:32-");
    if (I != StringRef::npos)
      Res = (Ref.take_front(I) + "-f80:128-" + Ref.drop_front(I + 8)).str();
  }

  return Res;
}

void llvm::UpgradeAttributes(AttrBuilder &B) {
  StringRef FramePointer;
  Attribute A = B.getAttribute("no-frame-pointer-elim");
  if (A.isValid()) {
    // The value can be "true" or "false".
    FramePointer = A.getValueAsString() == "true" ? "all" : "none";
    B.removeAttribute("no-frame-pointer-elim");
  }
  if (B.contains("no-frame-pointer-elim-non-leaf")) {
    // The value is ignored. "no-frame-pointer-elim"="true" takes priority.
    if (FramePointer != "all")
      FramePointer = "non-leaf";
    B.removeAttribute("no-frame-pointer-elim-non-leaf");
  }
  if (!FramePointer.empty())
    B.addAttribute("frame-pointer", FramePointer);

  A = B.getAttribute("null-pointer-is-valid");
  if (A.isValid()) {
    // The value can be "true" or "false".
    bool NullPointerIsValid = A.getValueAsString() == "true";
    B.removeAttribute("null-pointer-is-valid");
    if (NullPointerIsValid)
      B.addAttribute(Attribute::NullPointerIsValid);
  }
}

void llvm::UpgradeOperandBundles(std::vector<OperandBundleDef> &Bundles) {
  // clang.arc.attachedcall bundles are now required to have an operand.
  // If they don't, it's okay to drop them entirely: when there is an operand,
  // the "attachedcall" is meaningful and required, but without an operand,
  // it's just a marker NOP.  Dropping it merely prevents an optimization.
  erase_if(Bundles, [&](OperandBundleDef &OBD) {
    return OBD.getTag() == "clang.arc.attachedcall" &&
           OBD.inputs().empty();
  });
}
