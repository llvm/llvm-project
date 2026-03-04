//===-- LoongArchTargetTransformInfo.cpp - LoongArch specific TTI ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a TargetTransformInfo analysis pass specific to the
/// LoongArch target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#include "LoongArchTargetTransformInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/CostTable.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGenTypes/MachineValueType.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/InstructionCost.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "loongarchtti"

struct CostKindCosts {
  unsigned LatencyCost = ~0U;
  unsigned RecipThroughputCost = ~0U;
  unsigned CodeSizeCost = ~0U;
  unsigned SizeAndLatencyCost = ~0U;

  std::optional<unsigned>
  operator[](TargetTransformInfo::TargetCostKind Kind) const {
    unsigned Cost = ~0U;
    switch (Kind) {
    case llvm::TargetTransformInfo::TCK_Latency:
      Cost = LatencyCost;
      break;
    case TargetTransformInfo::TCK_RecipThroughput:
      Cost = RecipThroughputCost;
      break;
    case TargetTransformInfo::TCK_CodeSize:
      Cost = CodeSizeCost;
      break;
    case TargetTransformInfo::TCK_SizeAndLatency:
      Cost = SizeAndLatencyCost;
      break;
    }
    if (Cost == ~0U)
      return std::nullopt;
    return Cost;
  }
};
using CostKindTblEntry = CostTblEntryT<CostKindCosts>;
using TypeConversionCostTblEntry = TypeConversionCostTblEntryT<CostKindCosts>;

TypeSize LoongArchTTIImpl::getRegisterBitWidth(
    TargetTransformInfo::RegisterKind K) const {
  TypeSize DefSize = TargetTransformInfoImplBase::getRegisterBitWidth(K);
  switch (K) {
  case TargetTransformInfo::RGK_Scalar:
    return TypeSize::getFixed(ST->is64Bit() ? 64 : 32);
  case TargetTransformInfo::RGK_FixedWidthVector:
    if (ST->hasExtLASX())
      return TypeSize::getFixed(256);
    if (ST->hasExtLSX())
      return TypeSize::getFixed(128);
    [[fallthrough]];
  case TargetTransformInfo::RGK_ScalableVector:
    return DefSize;
  }

  llvm_unreachable("Unsupported register kind");
}

unsigned LoongArchTTIImpl::getNumberOfRegisters(unsigned ClassID) const {
  switch (ClassID) {
  case LoongArchRegisterClass::GPRRC:
    // 30 = 32 GPRs - r0 (zero register) - r21 (non-allocatable)
    return 30;
  case LoongArchRegisterClass::FPRRC:
    return ST->hasBasicF() ? 32 : 0;
  case LoongArchRegisterClass::VRRC:
    return ST->hasExtLSX() ? 32 : 0;
  }
  llvm_unreachable("unknown register class");
}

unsigned LoongArchTTIImpl::getRegisterClassForType(bool Vector,
                                                   Type *Ty) const {
  if (Vector)
    return LoongArchRegisterClass::VRRC;
  if (!Ty)
    return LoongArchRegisterClass::GPRRC;

  Type *ScalarTy = Ty->getScalarType();
  if ((ScalarTy->isFloatTy() && ST->hasBasicF()) ||
      (ScalarTy->isDoubleTy() && ST->hasBasicD())) {
    return LoongArchRegisterClass::FPRRC;
  }

  return LoongArchRegisterClass::GPRRC;
}

unsigned LoongArchTTIImpl::getMaxInterleaveFactor(ElementCount VF) const {
  return ST->getMaxInterleaveFactor();
}

const char *LoongArchTTIImpl::getRegisterClassName(unsigned ClassID) const {
  switch (ClassID) {
  case LoongArchRegisterClass::GPRRC:
    return "LoongArch::GPRRC";
  case LoongArchRegisterClass::FPRRC:
    return "LoongArch::FPRRC";
  case LoongArchRegisterClass::VRRC:
    return "LoongArch::VRRC";
  }
  llvm_unreachable("unknown register class");
}

TargetTransformInfo::PopcntSupportKind
LoongArchTTIImpl::getPopcntSupport(unsigned TyWidth) const {
  assert(isPowerOf2_32(TyWidth) && "Ty width must be power of 2");
  return ST->hasExtLSX() ? TTI::PSK_FastHardware : TTI::PSK_Software;
}

unsigned LoongArchTTIImpl::getCacheLineSize() const { return 64; }

unsigned LoongArchTTIImpl::getPrefetchDistance() const { return 200; }

bool LoongArchTTIImpl::enableWritePrefetching() const { return true; }

bool LoongArchTTIImpl::shouldExpandReduction(const IntrinsicInst *II) const {
  switch (II->getIntrinsicID()) {
  default:
    return true;
  case Intrinsic::vector_reduce_add:
  case Intrinsic::vector_reduce_and:
  case Intrinsic::vector_reduce_or:
  case Intrinsic::vector_reduce_smax:
  case Intrinsic::vector_reduce_smin:
  case Intrinsic::vector_reduce_umax:
  case Intrinsic::vector_reduce_umin:
  case Intrinsic::vector_reduce_xor:
    return false;
  }
}

LoongArchTTIImpl::TTI::MemCmpExpansionOptions
LoongArchTTIImpl::enableMemCmpExpansion(bool OptSize, bool IsZeroCmp) const {
  TTI::MemCmpExpansionOptions Options;

  if (!ST->hasUAL())
    return Options;

  Options.MaxNumLoads = TLI->getMaxExpandSizeMemcmp(OptSize);
  Options.NumLoadsPerBlock = Options.MaxNumLoads;
  Options.AllowOverlappingLoads = true;

  // TODO: Support for vectors.
  if (ST->is64Bit()) {
    Options.LoadSizes = {8, 4, 2, 1};
    Options.AllowedTailExpansions = {3, 5, 6};
  } else {
    Options.LoadSizes = {4, 2, 1};
    Options.AllowedTailExpansions = {3};
  }

  return Options;
}

InstructionCost LoongArchTTIImpl::getArithmeticInstrCost(
    unsigned Opcode, Type *Ty, TTI::TargetCostKind CostKind,
    TTI::OperandValueInfo Op1Info, TTI::OperandValueInfo Op2Info,
    ArrayRef<const Value *> Args, const Instruction *CxtI) const {

  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Ty);
  int ISD = TLI->InstructionOpcodeToISD(Opcode);

  // Vector multiply by pow2 will be simplified to shifts.
  // Vector multiply by -pow2 will be simplified to shifts/negates.
  if (ISD == ISD::MUL && Op2Info.isConstant() &&
      (Op2Info.isPowerOf2() || Op2Info.isNegatedPowerOf2())) {
    InstructionCost Cost =
        getArithmeticInstrCost(Instruction::Shl, Ty, CostKind,
                               Op1Info.getNoProps(), Op2Info.getNoProps());
    if (Op2Info.isNegatedPowerOf2())
      Cost += getArithmeticInstrCost(Instruction::Sub, Ty, CostKind);
    return Cost;
  }

  // On LoongArch, vector signed division by constants power-of-two are
  // normally expanded to the sequence SRA + SRL + ADD + SRA.
  // The OperandValue properties may not be the same as that of the previous
  // operation; conservatively assume OP_None.
  if ((ISD == ISD::SDIV || ISD == ISD::SREM) && Op2Info.isConstant() &&
      Op2Info.isPowerOf2()) {
    InstructionCost Cost =
        2 * getArithmeticInstrCost(Instruction::AShr, Ty, CostKind,
                                   Op1Info.getNoProps(), Op2Info.getNoProps());
    Cost += getArithmeticInstrCost(Instruction::LShr, Ty, CostKind,
                                   Op1Info.getNoProps(), Op2Info.getNoProps());
    Cost += getArithmeticInstrCost(Instruction::Add, Ty, CostKind,
                                   Op1Info.getNoProps(), Op2Info.getNoProps());

    if (ISD == ISD::SREM) {
      // For SREM: (X % C) is the equivalent of (X - (X/C)*C)
      Cost +=
          getArithmeticInstrCost(Instruction::Mul, Ty, CostKind,
                                 Op1Info.getNoProps(), Op2Info.getNoProps());
      Cost +=
          getArithmeticInstrCost(Instruction::Sub, Ty, CostKind,
                                 Op1Info.getNoProps(), Op2Info.getNoProps());
    }

    return Cost;
  }
  // Vector unsigned division/remainder will be simplified to shifts/masks.
  if ((ISD == ISD::UDIV || ISD == ISD::UREM) && Op2Info.isConstant() &&
      Op2Info.isPowerOf2()) {
    if (ISD == ISD::UDIV)
      return getArithmeticInstrCost(Instruction::LShr, Ty, CostKind,
                                    Op1Info.getNoProps(), Op2Info.getNoProps());
    // UREM
    return getArithmeticInstrCost(Instruction::And, Ty, CostKind,
                                  Op1Info.getNoProps(), Op2Info.getNoProps());
  }

  static const CostKindTblEntry LSXCostTable[]{
      {ISD::ADD, MVT::v16i8, {1, 1}}, // vaddi.b/vadd.b
      {ISD::ADD, MVT::v8i16, {1, 1}}, // vaddi.h/vadd.h
      {ISD::ADD, MVT::v4i32, {1, 1}}, // vaddi.w/vadd.w
      {ISD::ADD, MVT::v2i64, {1, 1}}, // vaddi.d/vadd.d

      {ISD::SUB, MVT::v16i8, {1, 1}}, // vsubi.b/vsub.b
      {ISD::SUB, MVT::v8i16, {1, 1}}, // vsubi.h/vsub.h
      {ISD::SUB, MVT::v4i32, {1, 1}}, // vsubi.w/vsub.w
      {ISD::SUB, MVT::v2i64, {1, 1}}, // vsubi.d/vsub.d

      {ISD::MUL, MVT::v16i8, {4, 2}}, // vmul.b
      {ISD::MUL, MVT::v8i16, {4, 2}}, // vmul.h
      {ISD::MUL, MVT::v4i32, {4, 2}}, // vmul.w
      {ISD::MUL, MVT::v2i64, {4, 2}}, // vmul.d

      {ISD::SDIV, MVT::v16i8, {38, 76}}, // vdiv.b
      {ISD::SDIV, MVT::v8i16, {24, 44}}, // vdiv.h
      {ISD::SDIV, MVT::v4i32, {17, 28}}, // vdiv.w
      {ISD::SDIV, MVT::v2i64, {14, 19}}, // vdiv.d

      {ISD::UDIV, MVT::v16i8, {38, 80}}, // vdiv.bu
      {ISD::UDIV, MVT::v8i16, {24, 44}}, // vdiv.hu
      {ISD::UDIV, MVT::v4i32, {17, 28}}, // vdiv.wu
      {ISD::UDIV, MVT::v2i64, {14, 19}}, // vdiv.du

      {ISD::SREM, MVT::v16i8, {38, 76}}, // vmod.b
      {ISD::SREM, MVT::v8i16, {24, 44}}, // vmod.h
      {ISD::SREM, MVT::v4i32, {17, 27}}, // vmod.w
      {ISD::SREM, MVT::v2i64, {14, 19}}, // vmod.d

      {ISD::UREM, MVT::v16i8, {38, 80}}, // vmod.bu
      {ISD::UREM, MVT::v8i16, {24, 44}}, // vmod.hu
      {ISD::UREM, MVT::v4i32, {17, 28}}, // vmod.wu
      {ISD::UREM, MVT::v2i64, {14, 19}}, // vmod.du

      {ISD::SHL, MVT::v16i8, {1, 1}}, // vslli.b/vsll.b
      {ISD::SHL, MVT::v8i16, {1, 1}}, // vslli.h/vsll.h
      {ISD::SHL, MVT::v4i32, {1, 1}}, // vslli.w/vsll.w
      {ISD::SHL, MVT::v2i64, {1, 1}}, // vslli.d/vsll.d

      {ISD::SRL, MVT::v16i8, {1, 1}}, // vsrli.b/vsrl.b
      {ISD::SRL, MVT::v8i16, {1, 1}}, // vsrli.h/vsrl.h
      {ISD::SRL, MVT::v4i32, {1, 1}}, // vsrli.w/vsrl.w
      {ISD::SRL, MVT::v2i64, {1, 1}}, // vsrli.d/vsrl.d

      {ISD::SRA, MVT::v16i8, {1, 1}}, // vsrai.b/vsra.b
      {ISD::SRA, MVT::v8i16, {1, 1}}, // vsrai.h/vsra.h
      {ISD::SRA, MVT::v4i32, {1, 1}}, // vsrai.w/vsra.w
      {ISD::SRA, MVT::v2i64, {1, 1}}, // vsrai.d/vsra.d

      {ISD::AND, MVT::v16i8, {1, 1}}, // vand.b/vand.v
      {ISD::AND, MVT::v8i16, {1, 1}}, // vand.v
      {ISD::AND, MVT::v4i32, {1, 1}}, // vand.v
      {ISD::AND, MVT::v2i64, {1, 1}}, // vand.v

      {ISD::OR, MVT::v16i8, {1, 1}}, // vori.b/vor.v
      {ISD::OR, MVT::v8i16, {1, 1}}, // vor.v
      {ISD::OR, MVT::v4i32, {1, 1}}, // vor.v
      {ISD::OR, MVT::v2i64, {1, 1}}, // vor.v

      {ISD::XOR, MVT::v16i8, {1, 1}}, // vxori.b/vxor.v
      {ISD::XOR, MVT::v8i16, {1, 1}}, // vxor.v
      {ISD::XOR, MVT::v4i32, {1, 1}}, // vxor.v
      {ISD::XOR, MVT::v2i64, {1, 1}}, // vxor.v

      {ISD::FADD, MVT::v4f32, {3, 1}}, // vfadd.s
      {ISD::FADD, MVT::v2f64, {3, 1}}, // vfadd.d

      {ISD::FSUB, MVT::v4f32, {3, 1}}, // vfsub.s
      {ISD::FSUB, MVT::v2f64, {3, 1}}, // vfsub.d

      {ISD::FMUL, MVT::v4f32, {5, 2}}, // vfmul.s
      {ISD::FMUL, MVT::v2f64, {5, 2}}, // vfmul.d

      {ISD::FDIV, MVT::v4f32, {16, 26}}, // vfdiv.s
      {ISD::FDIV, MVT::v2f64, {12, 18}}, // vfdiv.d
  };

  if (ST->hasExtLSX()) {
    if (const auto *Entry = CostTableLookup(LSXCostTable, ISD, LT.second))
      if (auto KindCost = Entry->Cost[CostKind])
        return LT.first * *KindCost;
  }

  static const CostKindTblEntry LASXUniformConstCostTable[]{
      {ISD::ADD, MVT::v32i8, {1, 1}},  // xvaddi.b/xvadd.b
      {ISD::ADD, MVT::v16i16, {1, 1}}, // xvaddi.h/xvadd.h
      {ISD::ADD, MVT::v8i32, {1, 1}},  // xvaddi.w/xvadd.w
      {ISD::ADD, MVT::v4i64, {1, 1}},  // xvaddi.d/xvadd.d

      {ISD::SUB, MVT::v32i8, {1, 1}},  // xvsubi.b/xvsub.b
      {ISD::SUB, MVT::v16i16, {1, 1}}, // xvsubi.h/xvsub.h
      {ISD::SUB, MVT::v8i32, {1, 1}},  // xvsubi.w/xvsub.w
      {ISD::SUB, MVT::v4i64, {1, 1}},  // xvsubi.d/xvsub.d

      {ISD::MUL, MVT::v32i8, {4, 2}},  // xvmul.b
      {ISD::MUL, MVT::v16i16, {4, 2}}, // xvmul.h
      {ISD::MUL, MVT::v8i32, {4, 2}},  // xvmul.w
      {ISD::MUL, MVT::v4i64, {4, 2}},  // xvmul.d

      {ISD::SDIV, MVT::v32i8, {38, 76}},  // xvdiv.b
      {ISD::SDIV, MVT::v16i16, {24, 43}}, // xvdiv.h
      {ISD::SDIV, MVT::v8i32, {17, 28}},  // xvdiv.w
      {ISD::SDIV, MVT::v4i64, {14, 19}},  // xvdiv.d

      {ISD::UDIV, MVT::v32i8, {38, 76}},  // xvdiv.bu
      {ISD::UDIV, MVT::v16i16, {24, 43}}, // xvdiv.hu
      {ISD::UDIV, MVT::v8i32, {17, 28}},  // xvdiv.wu
      {ISD::UDIV, MVT::v4i64, {14, 19}},  // xvdiv.du

      {ISD::SREM, MVT::v32i8, {38, 76}},  // xvmod.b
      {ISD::SREM, MVT::v16i16, {24, 44}}, // xvmod.h
      {ISD::SREM, MVT::v8i32, {17, 28}},  // xvmod.w
      {ISD::SREM, MVT::v4i64, {14, 19}},  // xvmod.d

      {ISD::UREM, MVT::v32i8, {38, 76}},  // xvmod.bu
      {ISD::UREM, MVT::v16i16, {24, 43}}, // xvmod.hu
      {ISD::UREM, MVT::v8i32, {17, 28}},  // xvmod.wu
      {ISD::UREM, MVT::v4i64, {14, 19}},  // xvmod.du

      {ISD::SHL, MVT::v32i8, {1, 1}},  // xvslli.b/xvsll.b
      {ISD::SHL, MVT::v16i16, {1, 1}}, // xvslli.h/xvsll.h
      {ISD::SHL, MVT::v8i32, {1, 1}},  // xvslli.w/xvsll.w
      {ISD::SHL, MVT::v4i64, {1, 1}},  // xvslli.d/xvsll.d

      {ISD::SRL, MVT::v32i8, {1, 1}},  // xvsrli.b/xvsrl.b
      {ISD::SRL, MVT::v16i16, {1, 1}}, // xvsrli.h/xvsrl.h
      {ISD::SRL, MVT::v8i32, {1, 1}},  // xvsrli.w/xvsrl.w
      {ISD::SRL, MVT::v4i64, {1, 1}},  // xvsrli.d/xvsrl.d

      {ISD::SRA, MVT::v32i8, {1, 1}},  // xvsrai.b/xvsra.b
      {ISD::SRA, MVT::v16i16, {1, 1}}, // xvsrai.h/xvsra.h
      {ISD::SRA, MVT::v8i32, {1, 1}},  // xvsrai.w/xvsra.w
      {ISD::SRA, MVT::v4i64, {1, 1}},  // xvsrai.d/xvsra.d

      {ISD::AND, MVT::v32i8, {1, 1}},  // xvandi.b/xvand.v
      {ISD::AND, MVT::v16i16, {1, 1}}, // xvand.v
      {ISD::AND, MVT::v8i32, {1, 1}},  // xvand.v
      {ISD::AND, MVT::v4i64, {1, 1}},  // xvand.v

      {ISD::OR, MVT::v32i8, {1, 1}},  // xvori.b/xvor.v
      {ISD::OR, MVT::v16i16, {1, 1}}, // xvor.v
      {ISD::OR, MVT::v8i32, {1, 1}},  // xvor.v
      {ISD::OR, MVT::v4i64, {1, 1}},  // xvor.v

      {ISD::XOR, MVT::v32i8, {1, 1}},  // xvxori.b/xvxor.v
      {ISD::XOR, MVT::v16i16, {1, 1}}, // xvxor.v
      {ISD::XOR, MVT::v8i32, {1, 1}},  // xvxor.v
      {ISD::XOR, MVT::v4i64, {1, 1}},  // xvxor.v

      {ISD::FADD, MVT::v8f32, {3, 1}}, // xvfadd.s
      {ISD::FADD, MVT::v4f64, {3, 1}}, // xvfadd.d

      {ISD::FSUB, MVT::v8f32, {3, 1}}, // xvfsub.s
      {ISD::FSUB, MVT::v4f64, {3, 1}}, // xvfsub.d

      {ISD::FMUL, MVT::v8f32, {5, 2}}, // xvfmul.s
      {ISD::FMUL, MVT::v4f64, {5, 2}}, // xvfmul.d

      {ISD::FDIV, MVT::v8f32, {15, 26}}, // xvfdiv.s
      {ISD::FDIV, MVT::v4f64, {12, 18}}, // xvfdiv.d
  };

  if (ST->hasExtLASX()) {
    if (const auto *Entry =
            CostTableLookup(LASXUniformConstCostTable, ISD, LT.second))
      if (auto KindCost = Entry->Cost[CostKind])
        return LT.first * *KindCost;
  }

  static const CostKindTblEntry LA64CostTable[]{
      {ISD::ADD, MVT::i64, {1, 1}}, // addi.d/add.d
      {ISD::SUB, MVT::i64, {1, 1}}, // subi.d/sub.d
      {ISD::MUL, MVT::i64, {4, 2}}, // mul.d

      {ISD::SDIV, MVT::i64, {18, 26}}, // div.d
      {ISD::UDIV, MVT::i64, {18, 26}}, // div.du
      {ISD::SREM, MVT::i64, {18, 26}}, // mod.d
      {ISD::UREM, MVT::i64, {18, 26}}, // mod.du

      {ISD::SHL, MVT::i64, {1, 1}}, // slli.d/sll.d
      {ISD::SRL, MVT::i64, {1, 1}}, // srli.d/srl.d
      {ISD::SRA, MVT::i64, {1, 1}}, // srai.d/sra.d

      {ISD::AND, MVT::i64, {1, 1}}, // andi.d/and.d
      {ISD::OR, MVT::i64, {1, 1}},  // ori.d/or.d
      {ISD::XOR, MVT::i64, {1, 1}}, // xori.d/xor.d

      {ISD::FADD, MVT::f64, {3, 1}},  // fadd.d
      {ISD::FSUB, MVT::f64, {3, 1}},  // fsub.d
      {ISD::FMUL, MVT::f64, {5, 2}},  // fmul.d
      {ISD::FDIV, MVT::f64, {12, 9}}, // fdiv.d
  };

  if (ST->is64Bit()) {
    if (const auto *Entry = CostTableLookup(LA64CostTable, ISD, LT.second))
      if (auto KindCost = Entry->Cost[CostKind])
        return LT.first * *KindCost;
  }

  static const CostKindTblEntry LA32CostTable[]{
      {ISD::ADD, MVT::i32, {1, 1}}, // addi.w/add.w
      {ISD::SUB, MVT::i32, {1, 1}}, // subi.w/sub.w
      {ISD::MUL, MVT::i32, {4, 2}}, // mul.w

      {ISD::SDIV, MVT::i32, {11, 24}}, // div.w
      {ISD::UDIV, MVT::i32, {12, 24}}, // div.wu
      {ISD::SREM, MVT::i32, {11, 24}}, // mod.w
      {ISD::UREM, MVT::i32, {12, 24}}, // mod.wu

      {ISD::SHL, MVT::i32, {1, 1}}, // slli.w/sll.w
      {ISD::SRL, MVT::i32, {1, 1}}, // srli.w/srl.w
      {ISD::SRA, MVT::i32, {1, 1}}, // srai.w/sra.w

      {ISD::AND, MVT::i32, {1, 1}}, // andi.w/and.w
      {ISD::OR, MVT::i32, {1, 1}},  // ori.w/or.w
      {ISD::XOR, MVT::i32, {1, 1}}, // xori.w/xor.w

      {ISD::FADD, MVT::f32, {3, 1}}, // fadd.s
      {ISD::FSUB, MVT::f32, {3, 1}}, // fsub.s
      {ISD::FMUL, MVT::f32, {5, 2}}, // fmul.s
      {ISD::FDIV, MVT::f32, {9, 8}}, // fdiv.s
  };

  if (const auto *Entry = CostTableLookup(LA32CostTable, ISD, LT.second))
    if (auto KindCost = Entry->Cost[CostKind])
      return LT.first * *KindCost;

  // Fallback to the default implementation.
  return BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, Op1Info, Op2Info,
                                       Args, CxtI);
}

InstructionCost LoongArchTTIImpl::getVectorInstrCost(
    unsigned Opcode, Type *Val, TTI::TargetCostKind CostKind, unsigned Index,
    const Value *Op0, const Value *Op1) const {

  assert(Val->isVectorTy() && "This must be a vector type");

  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Val);
  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  InstructionCost RegisterFileMoveCost = 0;

  static const CostKindTblEntry CostTable[]{
      {ISD::EXTRACT_VECTOR_ELT, MVT::i8, {3, 4}},  // vpickve2gr.b
      {ISD::EXTRACT_VECTOR_ELT, MVT::i16, {3, 4}}, // vpickve2gr.h
      {ISD::EXTRACT_VECTOR_ELT, MVT::i32, {3, 4}}, // vpickve2gr.w
      {ISD::EXTRACT_VECTOR_ELT, MVT::i64, {3, 4}}, // vpickve2gr.d

      {ISD::EXTRACT_VECTOR_ELT, MVT::f32, {1, 1}}, // vreplvei.w
      {ISD::EXTRACT_VECTOR_ELT, MVT::f64, {1, 1}}, // vreplvei.d
  };

  if (Index != -1U &&
      (ISD == ISD::EXTRACT_VECTOR_ELT || ISD == ISD::INSERT_VECTOR_ELT)) {

    if (!LT.second.isVector())
      return TTI::TCC_Free;

    unsigned SizeInBits = LT.second.getSizeInBits();
    unsigned NumElts = LT.second.getVectorNumElements();
    Index = Index % NumElts;

    if (SizeInBits > 128 && Index >= NumElts / 2 && !Val->isFPOrFPVectorTy()) {
      RegisterFileMoveCost += (ISD == ISD::INSERT_VECTOR_ELT ? 2 : 1);
    }

    if (ISD == ISD::INSERT_VECTOR_ELT) {
      // vldi/vrepli
      if (isa_and_nonnull<PoisonValue>(Op0) && isa_and_nonnull<Constant>(Op1)) {
        return 1 + RegisterFileMoveCost;
      }

      // vldi + vextrins
      if (isa_and_nonnull<ConstantFP>(Op1)) {
        return 2 + RegisterFileMoveCost;
      }

      // vextrins
      if (Op1 &&
          (Op1->getType()->isFloatTy() || Op1->getType()->isDoubleTy())) {
        return 1 + RegisterFileMoveCost;
      }

      // vinsgr2vr
      if (CostKind == TTI::TCK_RecipThroughput) {
        return 4 + RegisterFileMoveCost;
      }
      if (CostKind == TTI::TCK_Latency) {
        return 3 + RegisterFileMoveCost;
      }
    }

    if (auto *Entry =
            CostTableLookup(CostTable, ISD, LT.second.getScalarType()))
      if (auto KindCost = Entry->Cost[CostKind])
        return *KindCost + RegisterFileMoveCost;
  }

  return BaseT::getVectorInstrCost(Opcode, Val, CostKind, Index, Op0, Op1) +
         RegisterFileMoveCost;
}

InstructionCost LoongArchTTIImpl::getMemoryOpCost(unsigned Opcode, Type *Src,
                                                  Align Alignment,
                                                  unsigned AddressSpace,
                                                  TTI::TargetCostKind CostKind,
                                                  TTI::OperandValueInfo OpInfo,
                                                  const Instruction *I) const {

  // Legalize the type.
  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Src);

  switch (CostKind) {
  default:
    return BaseT::getMemoryOpCost(Opcode, Src, Alignment, AddressSpace,
                                  CostKind, OpInfo, I);
  case TTI::TCK_RecipThroughput:
    return 2 * LT.first;
  case TTI::TCK_Latency:
    unsigned Cost = 4;
    if (Src->isFloatingPointTy() || Src->isVectorTy()) {
      Cost += 1;
    }
    return Cost * LT.first;
  }
}

InstructionCost LoongArchTTIImpl::getCmpSelInstrCost(
    unsigned Opcode, Type *ValTy, Type *CondTy, CmpInst::Predicate VecPred,
    TTI::TargetCostKind CostKind, TTI::OperandValueInfo Op1Info,
    TTI::OperandValueInfo Op2Info, const Instruction *I) const {

  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(ValTy);
  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  MVT MTy = LT.second;

  InstructionCost ExtraCost = 0;

  // [x]vsl{t/e}[i] needs extra cost
  if (MTy.isVector() && MTy.getScalarSizeInBits() == 64 &&
      CostKind == TTI::TCK_Latency)
    if (VecPred != CmpInst::ICMP_EQ && VecPred != CmpInst::ICMP_NE)
      ExtraCost = 1;

  static const CostKindTblEntry LSXCostTable[] = {
      {ISD::SETCC, MVT::v16i8, {1, 1}}, // veq.b/...
      {ISD::SETCC, MVT::v8i16, {1, 1}}, // veq.h/...
      {ISD::SETCC, MVT::v4i32, {1, 1}}, // veq.w/...
      {ISD::SETCC, MVT::v2i64, {1, 1}}, // veq.d/...

      {ISD::SETCC, MVT::v4f32, {2, 1}}, // vfcmp.cond.s
      {ISD::SETCC, MVT::v2f64, {2, 1}}, // vfcmp.cond.d

      {ISD::SELECT, MVT::v16i8, {1, 2}}, // vbitsel.v
      {ISD::SELECT, MVT::v8i16, {1, 2}}, // vbitsel.v
      {ISD::SELECT, MVT::v4i32, {1, 2}}, // vbitsel.v
      {ISD::SELECT, MVT::v2i64, {1, 2}}, // vbitsel.v

      {ISD::SELECT, MVT::v4f32, {1, 2}}, // vbitsel.v
      {ISD::SELECT, MVT::v2f64, {1, 2}}, // vbitsel.v
  };

  static const CostKindTblEntry LASXCostTable[] = {
      {ISD::SETCC, MVT::v32i8, {1, 1}},  // xveq.b/...
      {ISD::SETCC, MVT::v16i16, {1, 1}}, // xveq.h/...
      {ISD::SETCC, MVT::v8i32, {1, 1}},  // xveq.w/...
      {ISD::SETCC, MVT::v4i64, {1, 1}},  // xveq.d/...

      {ISD::SETCC, MVT::v2f32, {2, 1}}, // xvfcmp.cond.s
      {ISD::SETCC, MVT::v4f64, {2, 1}}, // xvfcmp.cond.d

      {ISD::SELECT, MVT::v32i8, {1, 2}},  // xvbitsel.v
      {ISD::SELECT, MVT::v16i16, {1, 2}}, // xvbitsel.v
      {ISD::SELECT, MVT::v8i32, {1, 2}},  // xvbitsel.v
      {ISD::SELECT, MVT::v4i64, {1, 2}},  // xvbitsel.v

      {ISD::SELECT, MVT::v8f32, {1, 2}}, // xvbitsel.v
      {ISD::SELECT, MVT::v4f64, {1, 2}}, // xvbitsel.v
  };

  if (ST->hasExtLSX()) {
    if (const auto *Entry = CostTableLookup(LSXCostTable, ISD, MTy))
      if (auto KindCost = Entry->Cost[CostKind])
        return LT.first * (ExtraCost + *KindCost);
  }

  if (ST->hasExtLASX()) {
    if (const auto *Entry = CostTableLookup(LASXCostTable, ISD, MTy))
      if (auto KindCost = Entry->Cost[CostKind])
        return LT.first * (ExtraCost + *KindCost);
  }

  return BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy, VecPred, CostKind,
                                   Op1Info, Op2Info, I);
}

InstructionCost LoongArchTTIImpl::getCFInstrCost(unsigned Opcode,
                                                 TTI::TargetCostKind CostKind,
                                                 const Instruction *I) const {
  if (Opcode == Instruction::PHI) {
    return 0;
  }

  // Branches are assumed to be predicted.
  if (CostKind == TTI::TCK_RecipThroughput) {
    return 4;
  }
  return 1;
}

InstructionCost LoongArchTTIImpl::getShuffleCost(
    TTI::ShuffleKind Kind, VectorType *DstTy, VectorType *SrcTy,
    ArrayRef<int> Mask, TTI::TargetCostKind CostKind, int Index,
    VectorType *SubTp, ArrayRef<const Value *> Args,
    const Instruction *CxtI) const {

  // 64-bit packed float vectors (v2f32) are widened to type v4f32.
  // 64-bit packed integer vectors (v2i32) are widened to type v4i32.
  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(SrcTy);

  Kind = improveShuffleKindFromMask(Kind, Mask, SrcTy, Index, SubTp);

  if (Kind == TTI::SK_Broadcast) {
    // For Broadcasts we are splatting the first element from the first input
    // register, so only need to reference that input and all the output
    // registers are the same.
    LT.first = 1;

    // If we're broadcasting a load with [X]VLDREPL can do this for free.
    using namespace PatternMatch;
    if (!Args.empty() && match(Args[0], m_OneUse(m_Load(m_Value()))) &&
        (ST->hasExtLSX() || ST->hasExtLASX()))
      return TTI::TCC_Free;
  }

  // Attempt to detect a cheaper inlane shuffle, avoiding 128-bit subvector
  // permutation.
  bool IsInLaneShuffle = false;
  if (SrcTy->getPrimitiveSizeInBits() > 0 &&
      (SrcTy->getPrimitiveSizeInBits() % 128) == 0 &&
      SrcTy->getScalarSizeInBits() == LT.second.getScalarSizeInBits() &&
      Mask.size() == SrcTy->getElementCount().getKnownMinValue()) {
    unsigned NumLanes = SrcTy->getPrimitiveSizeInBits() / 128;
    unsigned NumEltsPerLane = Mask.size() / NumLanes;
    if ((Mask.size() % NumLanes) == 0) {
      IsInLaneShuffle = all_of(enumerate(Mask), [&](const auto &P) {
        return P.value() == PoisonMaskElem ||
               ((P.value() % Mask.size()) / NumEltsPerLane) ==
                   (P.index() / NumEltsPerLane);
      });
    }
  }

  // Subvector extractions are free if they start at the beginning of a
  // vector and cheap if the subvectors are aligned.
  if (Kind == TTI::SK_ExtractSubvector && LT.second.isVector()) {
    int NumElts = LT.second.getVectorNumElements();
    if ((Index % NumElts) == 0)
      return TTI::TCC_Free;
    std::pair<InstructionCost, MVT> SubLT = getTypeLegalizationCost(SubTp);
    if (SubLT.second.isVector()) {
      int NumSubElts = SubLT.second.getVectorNumElements();
      if ((Index % NumSubElts) == 0 && (NumElts % NumSubElts) == 0)
        return SubLT.first;
    }
    // If the extract subvector is not optimal, treat it as single op shuffle.
    Kind = TTI::SK_PermuteSingleSrc;
  }

  // Subvector insertions are cheap if the subvectors are aligned.
  // Note that in general, the insertion starting at the beginning of a vector
  // isn't free, because we need to preserve the rest of the wide vector,
  // but if the destination vector legalizes to the same width as the subvector
  // then the insertion will simplify to a (free) register copy.
  if (Kind == TTI::SK_InsertSubvector && LT.second.isVector()) {
    std::pair<InstructionCost, MVT> DstLT = getTypeLegalizationCost(DstTy);
    int NumElts = DstLT.second.getVectorNumElements();
    std::pair<InstructionCost, MVT> SubLT = getTypeLegalizationCost(SubTp);
    if (SubLT.second.isVector()) {
      int NumSubElts = SubLT.second.getVectorNumElements();
      bool MatchingTypes =
          NumElts == NumSubElts &&
          (SubTp->getElementCount().getKnownMinValue() % NumSubElts) == 0;
      if ((Index % NumSubElts) == 0 && (NumElts % NumSubElts) == 0)
        return MatchingTypes ? TTI::TCC_Free : SubLT.first;
    }

    // Attempt to match vextrins/xvinsve0 pattern.
    if (LT.first == 1 && SubLT.first == 1) {
      // vextrins.{w/d}
      if (ST->hasExtLSX() &&
          ((LT.second == MVT::v4f32 && SubLT.second == MVT::f32) ||
           (LT.second == MVT::v2f64 && SubLT.second == MVT::f64)))
        return 1;

      // xvinsve0.{w/d}
      if (ST->hasExtLASX() &&
          ((LT.second == MVT::v8f32 && SubLT.second == MVT::f32) ||
           (LT.second == MVT::v4f64 && SubLT.second == MVT::f64)))
        return 1;
    }

    // If the insertion is the lowest subvector then it will be blended
    // otherwise treat it like a 2-op shuffle.
    Kind =
        (Index == 0 && LT.first == 1) ? TTI::SK_Select : TTI::SK_PermuteTwoSrc;
  }

  static const CostKindTblEntry LSXCostTable[] = {
      {TTI::SK_Broadcast, MVT::v16i8, {1, 1}}, // vreplvei.b
      {TTI::SK_Broadcast, MVT::v8i16, {1, 1}}, // vreplvei.h
      {TTI::SK_Broadcast, MVT::v4i32, {1, 1}}, // vreplvei.w
      {TTI::SK_Broadcast, MVT::v2i64, {1, 1}}, // vreplvei.d
      {TTI::SK_Broadcast, MVT::v4f32, {1, 1}}, // vreplvei.w
      {TTI::SK_Broadcast, MVT::v2f64, {1, 1}}, // vreplvei.d

      {TTI::SK_Reverse, MVT::v16i8, {2, 2}}, // vshuf4i.w + vshuf4i.b
      {TTI::SK_Reverse, MVT::v8i16, {2, 2}}, // vshuf4i.d + vshuf4i.h
      {TTI::SK_Reverse, MVT::v4i32, {1, 1}}, // vshuf4i.w
      {TTI::SK_Reverse, MVT::v2i64, {1, 1}}, // vshuf4i.d
      {TTI::SK_Reverse, MVT::v4f32, {1, 1}}, // vshuf4i.w
      {TTI::SK_Reverse, MVT::v2f64, {1, 1}}, // vshuf4i.d

      {TTI::SK_Select, MVT::v16i8, {1, 2}}, // vbitsel.v
      {TTI::SK_Select, MVT::v8i16, {1, 2}}, // vbitsel.v
      {TTI::SK_Select, MVT::v4i32, {1, 2}}, // vbitsel.v
      {TTI::SK_Select, MVT::v2i64, {1, 1}}, // vshuf4i.d
      {TTI::SK_Select, MVT::v4f32, {1, 2}}, // vbitsel.v
      {TTI::SK_Select, MVT::v2f64, {1, 1}}, // vshuf4i.d

      {TTI::SK_Splice, MVT::v16i8, {3, 3}}, // vbsrl.v + vbsll.v + vor.v
      {TTI::SK_Splice, MVT::v8i16, {3, 3}}, // vbsrl.v + vbsll.v + vor.v
      {TTI::SK_Splice, MVT::v4i32, {3, 3}}, // vbsrl.v/vbsll.v + vor.v
      {TTI::SK_Splice, MVT::v2i64, {1, 1}}, // vshuf4i.d
      {TTI::SK_Splice, MVT::v4f32, {2, 2}}, // vbsrl.v/vbsll.v + vor.v
      {TTI::SK_Splice, MVT::v2f64, {1, 1}}, // vshuf4i.d

      {TTI::SK_Transpose, MVT::v16i8, {1, 1}}, // vpackev.b
      {TTI::SK_Transpose, MVT::v8i16, {1, 1}}, // vpackev.h
      {TTI::SK_Transpose, MVT::v4i32, {1, 1}}, // vpackev.w
      {TTI::SK_Transpose, MVT::v2i64, {1, 1}}, // vpackev.d
      {TTI::SK_Transpose, MVT::v4f32, {1, 1}}, // vpackev.w
      {TTI::SK_Transpose, MVT::v2f64, {1, 1}}, // vpackev.d

      {TTI::SK_PermuteSingleSrc, MVT::v16i8, {1, 2}}, // vshuf.b
      {TTI::SK_PermuteSingleSrc, MVT::v8i16, {1, 2}}, // vshuf.h
      {TTI::SK_PermuteSingleSrc, MVT::v4i32, {1, 1}}, // vshuf4i.w
      {TTI::SK_PermuteSingleSrc, MVT::v2i64, {1, 1}}, // vshuf4i.d
      {TTI::SK_PermuteSingleSrc, MVT::v4f32, {1, 1}}, // vshuf4i.w
      {TTI::SK_PermuteSingleSrc, MVT::v2f64, {1, 1}}, // vshuf4i.d

      {TTI::SK_PermuteTwoSrc, MVT::v16i8, {1, 2}}, // vshuf.b
      {TTI::SK_PermuteTwoSrc, MVT::v8i16, {1, 2}}, // vshuf.h
      {TTI::SK_PermuteTwoSrc, MVT::v4i32, {1, 2}}, // vshuf.w
      {TTI::SK_PermuteTwoSrc, MVT::v2i64, {1, 1}}, // vshuf4i.d
      {TTI::SK_PermuteTwoSrc, MVT::v4f32, {1, 2}}, // vshuf.w
      {TTI::SK_PermuteTwoSrc, MVT::v2f64, {1, 1}}, // vshuf4i.d
  };

  if (ST->hasExtLSX()) {
    if (const auto *Entry = CostTableLookup(LSXCostTable, Kind, LT.second))
      if (auto KindCost = Entry->Cost[CostKind])
        return LT.first * *KindCost;
  }

  static const CostKindTblEntry LASXInLaneCostTable[] = {
      {TTI::SK_PermuteSingleSrc, MVT::v32i8, {1, 2}},  // xvshuf.b
      {TTI::SK_PermuteSingleSrc, MVT::v16i16, {1, 2}}, // xvshuf.h

      {TTI::SK_PermuteTwoSrc, MVT::v32i8, {1, 2}},  // xvshuf.b
      {TTI::SK_PermuteTwoSrc, MVT::v16i16, {1, 2}}, // xvshuf.h
      {TTI::SK_PermuteTwoSrc, MVT::v8i32, {1, 2}},  // xvshuf.w
      {TTI::SK_PermuteTwoSrc, MVT::v4i64, {1, 2}},  // xvshuf.d
      {TTI::SK_PermuteTwoSrc, MVT::v8f32, {1, 2}},  // xvshuf.w
      {TTI::SK_PermuteTwoSrc, MVT::v4f64, {1, 2}},  // xvshuf.d
  };

  if (ST->hasExtLASX() && IsInLaneShuffle) {
    if (const auto *Entry =
            CostTableLookup(LASXInLaneCostTable, Kind, LT.second))
      if (auto KindCost = Entry->Cost[CostKind])
        return LT.first * *KindCost;
  }

  static const CostKindTblEntry LASXCostTable[] = {
      {TTI::SK_Broadcast, MVT::v32i8, {4, 2}},  // xvpermi.d + xvrepl128vei.b
      {TTI::SK_Broadcast, MVT::v16i16, {4, 2}}, // xvpermi.d + xvrepl128vei.h
      {TTI::SK_Broadcast, MVT::v8i32, {4, 2}},  // xvpermi.d + xvrepl128vei.w
      {TTI::SK_Broadcast, MVT::v4i64, {3, 1}},  // xvpermi.d
      {TTI::SK_Broadcast, MVT::v8f32, {4, 2}},  // xvpermi.d + xvrepl128vei.w
      {TTI::SK_Broadcast, MVT::v4f64, {3, 1}},  // xvpermi.d

      {TTI::SK_Reverse, MVT::v32i8, {5, 3}},  // xvpermi.d + xvshuf4i.w
                                              // + xvshuf4i.b
      {TTI::SK_Reverse, MVT::v16i16, {4, 2}}, // xvpermi.d + xvshuf4i.h
      {TTI::SK_Reverse, MVT::v8i32, {4, 2}},  // xvpermi.d + xvshuf4i.w
      {TTI::SK_Reverse, MVT::v4i64, {3, 1}},  // xvpermi.d
      {TTI::SK_Reverse, MVT::v8f32, {4, 2}},  // xvpermi.d + xvshuf4i.w
      {TTI::SK_Reverse, MVT::v4f64, {3, 1}},  // xvpermi.d

      {TTI::SK_Select, MVT::v32i8, {1, 2}},  // xvbitsel.v
      {TTI::SK_Select, MVT::v16i16, {1, 2}}, // xvbitsel.v
      {TTI::SK_Select, MVT::v8i32, {1, 2}},  // xvbitsel.v
      {TTI::SK_Select, MVT::v4i64, {1, 2}},  // xvbitsel.v
      {TTI::SK_Select, MVT::v8f32, {1, 2}},  // xvbitsel.v
      {TTI::SK_Select, MVT::v4f64, {1, 2}},  // xvbitsel.v

      {TTI::SK_Splice, MVT::v32i8, {6, 4}},  // xvpermi.q + xvbsll.v + xvbsrl.v
                                             // + xvor.v
      {TTI::SK_Splice, MVT::v16i16, {6, 4}}, // xvpermi.q + xvbsll.v + xvbsrl.v
                                             // + xvor.v
      {TTI::SK_Splice, MVT::v8i32, {6, 4}},  // xvpermi.q + xvbsll.v + xvbsrl.v
                                             // + xvor.v
      {TTI::SK_Splice, MVT::v4i64, {4, 2}},  // xvpermi.q + xvshuf4i.d
      {TTI::SK_Splice, MVT::v8f32, {6, 4}},  // xvpermi.q + xvbsll.v + xvbsrl.v
                                             // + xvor.v
      {TTI::SK_Splice, MVT::v4f64, {4, 2}},  // xvpermi.q + xvshuf4i.d

      {TTI::SK_Transpose, MVT::v32i8, {1, 1}},  // xvpackev.b
      {TTI::SK_Transpose, MVT::v16i16, {1, 1}}, // xvpackev.h
      {TTI::SK_Transpose, MVT::v8i32, {1, 1}},  // xvpackev.w
      {TTI::SK_Transpose, MVT::v4i64, {1, 1}},  // xvpackev.d
      {TTI::SK_Transpose, MVT::v8f32, {1, 1}},  // xvpackev.w
      {TTI::SK_Transpose, MVT::v4f64, {1, 1}},  // xvpackev.d

      {TTI::SK_PermuteSingleSrc, MVT::v32i8, {4, 3}},  // xvpermi.d + xvshuf.b
      {TTI::SK_PermuteSingleSrc, MVT::v16i16, {4, 3}}, // xvpermi.d + xvshuf.h
      {TTI::SK_PermuteSingleSrc, MVT::v8i32, {3, 1}},  // xvperm.w
      {TTI::SK_PermuteSingleSrc, MVT::v4i64, {3, 1}},  // xvpermi.d
      {TTI::SK_PermuteSingleSrc, MVT::v8f32, {3, 1}},  // xvperm.w
      {TTI::SK_PermuteSingleSrc, MVT::v4f64, {3, 1}},  // xvpermi.d

      {TTI::SK_PermuteTwoSrc, MVT::v32i8, {9, 8}},  // 2 *xvpermi.q + 2*xvshuf.b
                                                    // + xvbitsel.v
      {TTI::SK_PermuteTwoSrc, MVT::v16i16, {9, 8}}, // 2*xvpermi.q + 2*xvshuf.h
                                                    // + xvbitsel.v
      {TTI::SK_PermuteTwoSrc, MVT::v8i32, {7, 4}},  // 2*xvperm.w + xvbitsel.v
      {TTI::SK_PermuteTwoSrc, MVT::v4i64, {7, 4}},  // 2*xvpermi.d + xvshuf.d
      {TTI::SK_PermuteTwoSrc, MVT::v8f32, {7, 4}},  // 2*xvperm.w + xvbitsel.v
      {TTI::SK_PermuteTwoSrc, MVT::v4f64, {7, 4}},  // 2*xvpermi.d + xvshuf.d
  };

  if (ST->hasExtLASX()) {
    if (const auto *Entry = CostTableLookup(LASXCostTable, Kind, LT.second))
      if (auto KindCost = Entry->Cost[CostKind])
        return LT.first * *KindCost;
  }

  return BaseT::getShuffleCost(Kind, DstTy, SrcTy, Mask, CostKind, Index,
                               SubTp);
}

InstructionCost
LoongArchTTIImpl::getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                        TTI::TargetCostKind CostKind) const {
  static const CostKindTblEntry LASXCostTable[] = {
      {ISD::ABS, MVT::v32i8, {1, 2}},  // xvsigncov.b
      {ISD::ABS, MVT::v16i16, {1, 2}}, // xvsigncov.h
      {ISD::ABS, MVT::v8i32, {1, 2}},  // xvsigncov.w
      {ISD::ABS, MVT::v4i64, {1, 2}},  // xvsigncov.d

      {ISD::SADDSAT, MVT::v32i8, {1, 1}},  // xvsadd.b
      {ISD::SADDSAT, MVT::v16i16, {1, 1}}, // xvsadd.h
      {ISD::SADDSAT, MVT::v8i32, {1, 1}},  // xvsadd.w
      {ISD::SADDSAT, MVT::v4i64, {1, 1}},  // xvsadd.d

      {ISD::SSUBSAT, MVT::v32i8, {1, 1}},  // xvssub.b
      {ISD::SSUBSAT, MVT::v16i16, {1, 1}}, // xvssub.h
      {ISD::SSUBSAT, MVT::v8i32, {1, 1}},  // xvssub.w
      {ISD::SSUBSAT, MVT::v4i64, {1, 1}},  // xvssub.d

      {ISD::UADDSAT, MVT::v32i8, {1, 1}},  // xvadd.bu
      {ISD::UADDSAT, MVT::v16i16, {1, 1}}, // xvadd.hu
      {ISD::UADDSAT, MVT::v8i32, {1, 1}},  // xvadd.wu
      {ISD::UADDSAT, MVT::v4i64, {1, 1}},  // xvadd.du

      {ISD::USUBSAT, MVT::v32i8, {1, 1}},  // xvsub.bu
      {ISD::USUBSAT, MVT::v16i16, {1, 1}}, // xvsub.hu
      {ISD::USUBSAT, MVT::v8i32, {1, 1}},  // xvsub.wu
      {ISD::USUBSAT, MVT::v4i64, {1, 1}},  // xvsub.du

      {ISD::SMAX, MVT::v32i8, {1, 1}},  // xvmax.b
      {ISD::SMAX, MVT::v16i16, {1, 1}}, // xvmax.h
      {ISD::SMAX, MVT::v8i32, {1, 1}},  // xvmax.w
      {ISD::SMAX, MVT::v4i64, {2, 1}},  // xvmax.d

      {ISD::SMIN, MVT::v32i8, {1, 1}},  // xvmin.b
      {ISD::SMIN, MVT::v16i16, {1, 1}}, // xvmin.h
      {ISD::SMIN, MVT::v8i32, {1, 1}},  // xvmin.w
      {ISD::SMIN, MVT::v4i64, {2, 1}},  // xvmin.d

      {ISD::UMAX, MVT::v32i8, {1, 1}},  // xvmax.bu
      {ISD::UMAX, MVT::v16i16, {1, 1}}, // xvmax.hu
      {ISD::UMAX, MVT::v8i32, {1, 1}},  // xvmax.wu
      {ISD::UMAX, MVT::v4i64, {2, 1}},  // xvmax.du

      {ISD::UMIN, MVT::v32i8, {1, 1}},  // xvmin.bu
      {ISD::UMIN, MVT::v16i16, {1, 1}}, // xvmin.hu
      {ISD::UMIN, MVT::v8i32, {1, 1}},  // xvmin.wu
      {ISD::UMIN, MVT::v4i64, {2, 1}},  // xvmin.du

      {ISD::FMAXNUM, MVT::v8f32, {2, 1}}, // xvfmax.s
      {ISD::FMAXNUM, MVT::v4f64, {2, 1}}, // xvfmax.d
      {ISD::FMINNUM, MVT::v8f32, {2, 1}}, // xvfmin.s
      {ISD::FMINNUM, MVT::v4f64, {2, 1}}, // xvfmin.d

      {ISD::FLOG2, MVT::v8f32, {4, 1}}, // xvflogb.s
      {ISD::FLOG2, MVT::v4f64, {4, 1}}, // xvflogb.d

      {ISD::FMA, MVT::v8f32, {5, 2}}, // xvfmadd.s
      {ISD::FMA, MVT::v4f64, {5, 2}}, // xvfmadd.d

      {ISD::FSQRT, MVT::v8f32, {25, 28}}, // xvrsqrt.s
      {ISD::FSQRT, MVT::v4f64, {22, 20}}, // xvrsqrt.d

      {ISD::CTPOP, MVT::v32i8, {2, 2}},  // xvpcnt.b
      {ISD::CTPOP, MVT::v16i16, {2, 2}}, // xvpcnt.h
      {ISD::CTPOP, MVT::v8i32, {2, 2}},  // xvpcnt.w
      {ISD::CTPOP, MVT::v4i64, {2, 2}},  // xvpcnt.d

      {ISD::CTLZ, MVT::v32i8, {2, 1}},  // xvclz.b
      {ISD::CTLZ, MVT::v16i16, {2, 1}}, // xvclz.h
      {ISD::CTLZ, MVT::v8i32, {2, 1}},  // xvclz.w
      {ISD::CTLZ, MVT::v4i64, {2, 1}},  // xvclz.d

      {ISD::CTTZ, MVT::v32i8, {4, 4}},  // xvsubi.bu + xvandn.v + xvpcnt.b
      {ISD::CTTZ, MVT::v16i16, {4, 4}}, // xvsubi.hu + xvandn.v + xvpcnt.h
      {ISD::CTTZ, MVT::v8i32, {4, 4}},  // xvsubi.wu + xvandn.v + xvpcnt.w
      {ISD::CTTZ, MVT::v4i64, {4, 4}},  // xvsubi.du + xvandn.v + xvpcnt.d

      {ISD::BSWAP, MVT::v16i16, {1, 1}}, // xvshuf4i.b
      {ISD::BSWAP, MVT::v8i32, {1, 1}},  // xvshuf4i.b
      {ISD::BSWAP, MVT::v4i64, {2, 2}},  // xvshuf4i.b + xvshuf4i.w
  };

  static const CostKindTblEntry LSXCostTable[] = {
      {ISD::ABS, MVT::v16i8, {1, 2}}, // vsigncov.b
      {ISD::ABS, MVT::v8i16, {1, 2}}, // vsigncov.h
      {ISD::ABS, MVT::v4i32, {1, 2}}, // vsigncov.w
      {ISD::ABS, MVT::v2i64, {1, 2}}, // vsigncov.d

      {ISD::SADDSAT, MVT::v16i8, {1, 1}}, // vsadd.b
      {ISD::SADDSAT, MVT::v8i16, {1, 1}}, // vsadd.h
      {ISD::SADDSAT, MVT::v4i32, {1, 1}}, // vsadd.w
      {ISD::SADDSAT, MVT::v2i64, {1, 1}}, // vsadd.d

      {ISD::SSUBSAT, MVT::v16i8, {1, 1}}, // vssub.b
      {ISD::SSUBSAT, MVT::v8i16, {1, 1}}, // vssub.h
      {ISD::SSUBSAT, MVT::v4i32, {1, 1}}, // vssub.w
      {ISD::SSUBSAT, MVT::v2i64, {1, 1}}, // vssub.d

      {ISD::UADDSAT, MVT::v16i8, {1, 1}}, // vsadd.bu
      {ISD::UADDSAT, MVT::v8i16, {1, 1}}, // vsadd.hu
      {ISD::UADDSAT, MVT::v4i32, {1, 1}}, // vsadd.wu
      {ISD::UADDSAT, MVT::v2i64, {1, 1}}, // vsadd.du

      {ISD::USUBSAT, MVT::v16i8, {1, 1}}, // vssub.bu
      {ISD::USUBSAT, MVT::v8i16, {1, 1}}, // vssub.hu
      {ISD::USUBSAT, MVT::v4i32, {1, 1}}, // vssub.wu
      {ISD::USUBSAT, MVT::v2i64, {1, 1}}, // vssub.du

      {ISD::SMAX, MVT::v16i8, {1, 1}}, // vmax.b
      {ISD::SMAX, MVT::v8i16, {1, 1}}, // vmax.h
      {ISD::SMAX, MVT::v4i32, {1, 1}}, // vmax.w
      {ISD::SMAX, MVT::v2i64, {2, 1}}, // vmax.d

      {ISD::SMIN, MVT::v16i8, {1, 1}}, // vmin.b
      {ISD::SMIN, MVT::v8i16, {1, 1}}, // vmin.h
      {ISD::SMIN, MVT::v4i32, {1, 1}}, // vmin.w
      {ISD::SMIN, MVT::v2i64, {2, 1}}, // vmin.d

      {ISD::UMAX, MVT::v16i8, {1, 1}}, // vmax.bu
      {ISD::UMAX, MVT::v8i16, {1, 1}}, // vmax.hu
      {ISD::UMAX, MVT::v4i32, {1, 1}}, // vmax.wu
      {ISD::UMAX, MVT::v2i64, {2, 1}}, // vmax.du

      {ISD::UMIN, MVT::v16i8, {1, 1}}, // vmin.bu
      {ISD::UMIN, MVT::v8i16, {1, 1}}, // vmin.hu
      {ISD::UMIN, MVT::v4i32, {1, 1}}, // vmin.wu
      {ISD::UMIN, MVT::v2i64, {2, 1}}, // vmin.du

      {ISD::FMAXNUM, MVT::v4f32, {2, 1}}, // vfmax.s
      {ISD::FMAXNUM, MVT::v2f64, {2, 1}}, // vfmax.d
      {ISD::FMINNUM, MVT::v4f32, {2, 1}}, // vfmin.s
      {ISD::FMINNUM, MVT::v2f64, {2, 1}}, // vfmin.d

      {ISD::FLOG2, MVT::v4f32, {4, 1}}, // vflogb.s
      {ISD::FLOG2, MVT::v2f64, {4, 1}}, // vflogb.d

      {ISD::FMA, MVT::v4f32, {5, 2}}, // vfmadd.s
      {ISD::FMA, MVT::v2f64, {5, 2}}, // vfmadd.d

      {ISD::FSQRT, MVT::v4f32, {25, 28}}, // vfsqrt.s
      {ISD::FSQRT, MVT::v2f64, {22, 20}}, // vfsqrt.d

      {ISD::CTPOP, MVT::v16i8, {2, 2}}, // vpcnt.b
      {ISD::CTPOP, MVT::v8i16, {2, 2}}, // vpcnt.h
      {ISD::CTPOP, MVT::v4i32, {2, 2}}, // vpcnt.w
      {ISD::CTPOP, MVT::v2i64, {2, 2}}, // vpcnt.d

      {ISD::CTLZ, MVT::v16i8, {2, 1}}, // vclz.b
      {ISD::CTLZ, MVT::v8i16, {2, 1}}, // vclz.h
      {ISD::CTLZ, MVT::v4i32, {2, 1}}, // vclz.w
      {ISD::CTLZ, MVT::v2i64, {2, 1}}, // vclz.d

      {ISD::CTTZ, MVT::v16i8, {4, 4}}, // vsubi.bu + vandn.v + vpcnt.b
      {ISD::CTTZ, MVT::v8i16, {4, 4}}, // vsubi.hu + vandn.v + vpcnt.h
      {ISD::CTTZ, MVT::v4i32, {4, 4}}, // vsubi.wu + vandn.v + vpcnt.w
      {ISD::CTTZ, MVT::v2i64, {4, 4}}, // vsubi.du + vandn.v + vpcnt.d

      {ISD::BSWAP, MVT::v8i16, {1, 1}}, // vshuf4i.b
      {ISD::BSWAP, MVT::v4i32, {1, 1}}, // vshuf4i.b
      {ISD::BSWAP, MVT::v2i64, {2, 2}}, // vshuf4i.b + vshuf4i.w
  };

  static const CostKindTblEntry LA64CostTable[] = {
      {ISD::ABS, MVT::i8, {3, 3}},  // srai.d + xor + sub.d
      {ISD::ABS, MVT::i16, {3, 3}}, // srai.d + xor + sub.d
      {ISD::ABS, MVT::i32, {3, 3}}, // srai.d + xor + sub.d
      {ISD::ABS, MVT::i64, {3, 3}}, // srai.d + xor + sub.d

      {ISD::FMINNUM, MVT::f32, {2, 1}}, // fmin.s
      {ISD::FMINNUM, MVT::f64, {2, 1}}, // fmin.d

      {ISD::FLOG2, MVT::f32, {4, 1}}, // flogb.s
      {ISD::FLOG2, MVT::f64, {4, 1}}, // flogb.d

      {ISD::FMA, MVT::f32, {5, 2}}, // fmadd.s
      {ISD::FMA, MVT::f64, {5, 2}}, // fmadd.d

      {ISD::FSQRT, MVT::f32, {15, 9}},  // fsqrt.s
      {ISD::FSQRT, MVT::f64, {22, 10}}, // fsqrt.d

      {ISD::CTLZ, MVT::i8, {1, 1}},  // clz.b
      {ISD::CTLZ, MVT::i16, {1, 1}}, // clz.h
      {ISD::CTLZ, MVT::i32, {1, 1}}, // clz.w
      {ISD::CTLZ, MVT::i64, {1, 1}}, // clz.d

      {ISD::CTTZ, MVT::i8, {1, 1}},  // ctz.b
      {ISD::CTTZ, MVT::i16, {1, 1}}, // ctz.h
      {ISD::CTTZ, MVT::i32, {1, 1}}, // ctz.w
      {ISD::CTTZ, MVT::i64, {1, 1}}, // ctz.d

      {ISD::BITREVERSE, MVT::i8, {1, 1}},  // bitrev.4b
      {ISD::BITREVERSE, MVT::i16, {2, 2}}, // bitrev.d + srli.d
      {ISD::BITREVERSE, MVT::i32, {1, 1}}, // bitrev.w
      {ISD::BITREVERSE, MVT::i64, {1, 1}}, // bitrev.d

      {ISD::BSWAP, MVT::i16, {1, 1}}, // bswap.2h
      {ISD::BSWAP, MVT::i32, {1, 1}}, // bswap.2w
      {ISD::BSWAP, MVT::i64, {1, 1}}, // bswap.d
  };

  Type *RetTy = ICA.getReturnType();
  Type *OpTy = RetTy;
  Intrinsic::ID IID = ICA.getID();
  unsigned ISD = ISD::DELETED_NODE;
  switch (IID) {
  default:
    break;
  case Intrinsic::abs:
    ISD = ISD::ABS;
    break;
  case Intrinsic::sadd_sat:
    ISD = ISD::SADDSAT;
    break;
  case Intrinsic::ssub_sat:
    ISD = ISD::SSUBSAT;
    break;
  case Intrinsic::uadd_sat:
    ISD = ISD::UADDSAT;
    break;
  case Intrinsic::usub_sat:
    ISD = ISD::USUBSAT;
    break;
  case Intrinsic::smax:
    ISD = ISD::SMAX;
    break;
  case Intrinsic::smin:
    ISD = ISD::SMIN;
    break;
  case Intrinsic::umax:
    ISD = ISD::UMAX;
    break;
  case Intrinsic::umin:
    ISD = ISD::UMIN;
    break;
  case Intrinsic::maxnum:
    ISD = ISD::FMAXNUM;
    break;
  case Intrinsic::minnum:
    ISD = ISD::FMINNUM;
    break;
  case Intrinsic::log2:
    ISD = ISD::FLOG2;
    break;
  case Intrinsic::fma:
    ISD = ISD::FMA;
    break;
  case Intrinsic::sqrt:
    ISD = ISD::FSQRT;
    break;
  case Intrinsic::ctlz:
    ISD = ISD::CTLZ;
    break;
  case Intrinsic::ctpop:
    ISD = ISD::CTPOP;
    break;
  case Intrinsic::cttz:
    ISD = ISD::CTTZ;
    break;
  case Intrinsic::bitreverse:
    ISD = ISD::BITREVERSE;
    break;
  case Intrinsic::bswap:
    ISD = ISD::BSWAP;
    break;
  }

  if (ISD != ISD::DELETED_NODE) {

    std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(OpTy);
    MVT MTy = LT.second;

    if (ST->hasExtLASX())
      if (const auto *Entry = CostTableLookup(LASXCostTable, ISD, MTy))
        if (auto KindCost = Entry->Cost[CostKind])
          return LT.first * *KindCost;

    if (ST->hasExtLSX())
      if (const auto *Entry = CostTableLookup(LSXCostTable, ISD, MTy))
        if (auto KindCost = Entry->Cost[CostKind])
          return LT.first * *KindCost;

    if (ST->is64Bit())
      if (const auto *Entry = CostTableLookup(LA64CostTable, ISD, MTy))
        if (auto KindCost = Entry->Cost[CostKind])
          return LT.first * *KindCost;
  }

  return BaseT::getIntrinsicInstrCost(ICA, CostKind);
}

bool LoongArchTTIImpl::prefersVectorizedAddressing() const { return false; }

// TODO: Implement more hooks to provide TTI machinery for LoongArch.
