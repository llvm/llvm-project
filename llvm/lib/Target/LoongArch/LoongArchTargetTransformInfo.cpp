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

// TODO: Implement more hooks to provide TTI machinery for LoongArch.
