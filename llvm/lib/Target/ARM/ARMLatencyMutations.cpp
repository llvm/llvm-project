//===- ARMLatencyMutations.cpp - ARM Latency Mutations --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains the ARM definition DAG scheduling mutations which
/// change inter-instruction latencies
//
//===----------------------------------------------------------------------===//

#include "ARMLatencyMutations.h"
#include "ARMSubtarget.h"
#include "Thumb2InstrInfo.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleDAGMutation.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include <algorithm>
#include <array>
#include <initializer_list>
#include <memory>

namespace llvm {

namespace {

// Precompute information about opcodes to speed up pass

class InstructionInformation {
protected:
  struct IInfo {
    bool HasBRegAddr : 1;      // B-side of addr gen is a register
    bool HasBRegAddrShift : 1; // B-side of addr gen has a shift
    bool IsDivide : 1;         // Some form of integer divide
    bool IsInlineShiftALU : 1; // Inline shift+ALU
    bool IsMultiply : 1;       // Some form of integer multiply
    bool IsMVEIntMAC : 1;      // MVE 8/16/32-bit integer MAC operation
    bool IsNonSubwordLoad : 1; // Load which is a word or larger
    bool IsShift : 1;          // Shift operation
    bool IsRev : 1;            // REV operation
    bool ProducesQP : 1;       // Produces a vector register result
    bool ProducesDP : 1;       // Produces a double-precision register result
    bool ProducesSP : 1;       // Produces a single-precision register result
    bool ConsumesQP : 1;       // Consumes a vector register result
    bool ConsumesDP : 1;       // Consumes a double-precision register result
    bool ConsumesSP : 1;       // Consumes a single-precision register result
    unsigned MVEIntMACMatched; // Matched operand type (for MVE)
    unsigned AddressOpMask;    // Mask indicating which operands go into AGU
    IInfo()
        : HasBRegAddr(false), HasBRegAddrShift(false), IsDivide(false),
          IsInlineShiftALU(false), IsMultiply(false), IsMVEIntMAC(false),
          IsNonSubwordLoad(false), IsShift(false), IsRev(false),
          ProducesQP(false), ProducesDP(false), ProducesSP(false),
          ConsumesQP(false), ConsumesDP(false), ConsumesSP(false),
          MVEIntMACMatched(0), AddressOpMask(0) {}
  };
  typedef std::array<IInfo, ARM::INSTRUCTION_LIST_END> IInfoArray;
  IInfoArray Info;

public:
  // Always available information
  unsigned getAddressOpMask(unsigned Op) { return Info[Op].AddressOpMask; }
  bool hasBRegAddr(unsigned Op) { return Info[Op].HasBRegAddr; }
  bool hasBRegAddrShift(unsigned Op) { return Info[Op].HasBRegAddrShift; }
  bool isDivide(unsigned Op) { return Info[Op].IsDivide; }
  bool isInlineShiftALU(unsigned Op) { return Info[Op].IsInlineShiftALU; }
  bool isMultiply(unsigned Op) { return Info[Op].IsMultiply; }
  bool isMVEIntMAC(unsigned Op) { return Info[Op].IsMVEIntMAC; }
  bool isNonSubwordLoad(unsigned Op) { return Info[Op].IsNonSubwordLoad; }
  bool isRev(unsigned Op) { return Info[Op].IsRev; }
  bool isShift(unsigned Op) { return Info[Op].IsShift; }

  // information available if markDPConsumers is called.
  bool producesQP(unsigned Op) { return Info[Op].ProducesQP; }
  bool producesDP(unsigned Op) { return Info[Op].ProducesDP; }
  bool producesSP(unsigned Op) { return Info[Op].ProducesSP; }
  bool consumesQP(unsigned Op) { return Info[Op].ConsumesQP; }
  bool consumesDP(unsigned Op) { return Info[Op].ConsumesDP; }
  bool consumesSP(unsigned Op) { return Info[Op].ConsumesSP; }

  bool isMVEIntMACMatched(unsigned SrcOp, unsigned DstOp) {
    return SrcOp == DstOp || Info[DstOp].MVEIntMACMatched == SrcOp;
  }

  InstructionInformation(const ARMBaseInstrInfo *TII);

protected:
  void markDPProducersConsumers(const ARMBaseInstrInfo *TII);
};

InstructionInformation::InstructionInformation(const ARMBaseInstrInfo *TII) {
  using namespace ARM;

  std::initializer_list<unsigned> hasBRegAddrList = {
      t2LDRs, t2LDRBs, t2LDRHs, t2STRs, t2STRBs, t2STRHs,
      tLDRr,  tLDRBr,  tLDRHr,  tSTRr,  tSTRBr,  tSTRHr,
  };
  for (auto op : hasBRegAddrList) {
    Info[op].HasBRegAddr = true;
  }

  std::initializer_list<unsigned> hasBRegAddrShiftList = {
      t2LDRs, t2LDRBs, t2LDRHs, t2STRs, t2STRBs, t2STRHs,
  };
  for (auto op : hasBRegAddrShiftList) {
    Info[op].HasBRegAddrShift = true;
  }

  Info[t2SDIV].IsDivide = Info[t2UDIV].IsDivide = true;

  std::initializer_list<unsigned> isInlineShiftALUList = {
      t2ADCrs,  t2ADDSrs, t2ADDrs,  t2BICrs, t2EORrs,
      t2ORNrs,  t2RSBSrs, t2RSBrs,  t2SBCrs, t2SUBrs,
      t2SUBSrs, t2CMPrs,  t2CMNzrs, t2TEQrs, t2TSTrs,
  };
  for (auto op : isInlineShiftALUList) {
    Info[op].IsInlineShiftALU = true;
  }

  Info[t2SDIV].IsDivide = Info[t2UDIV].IsDivide = true;

  std::initializer_list<unsigned> isMultiplyList = {
      t2MUL,    t2MLA,     t2MLS,     t2SMLABB, t2SMLABT,  t2SMLAD,   t2SMLADX,
      t2SMLAL,  t2SMLALBB, t2SMLALBT, t2SMLALD, t2SMLALDX, t2SMLALTB, t2SMLALTT,
      t2SMLATB, t2SMLATT,  t2SMLAWT,  t2SMLSD,  t2SMLSDX,  t2SMLSLD,  t2SMLSLDX,
      t2SMMLA,  t2SMMLAR,  t2SMMLS,   t2SMMLSR, t2SMMUL,   t2SMMULR,  t2SMUAD,
      t2SMUADX, t2SMULBB,  t2SMULBT,  t2SMULL,  t2SMULTB,  t2SMULTT,  t2SMULWT,
      t2SMUSD,  t2SMUSDX,  t2UMAAL,   t2UMLAL,  t2UMULL,   tMUL,
  };
  for (auto op : isMultiplyList) {
    Info[op].IsMultiply = true;
  }

  std::initializer_list<unsigned> isMVEIntMACList = {
      MVE_VMLAS_qr_i16,    MVE_VMLAS_qr_i32,    MVE_VMLAS_qr_i8,
      MVE_VMLA_qr_i16,     MVE_VMLA_qr_i32,     MVE_VMLA_qr_i8,
      MVE_VQDMLAH_qrs16,   MVE_VQDMLAH_qrs32,   MVE_VQDMLAH_qrs8,
      MVE_VQDMLASH_qrs16,  MVE_VQDMLASH_qrs32,  MVE_VQDMLASH_qrs8,
      MVE_VQRDMLAH_qrs16,  MVE_VQRDMLAH_qrs32,  MVE_VQRDMLAH_qrs8,
      MVE_VQRDMLASH_qrs16, MVE_VQRDMLASH_qrs32, MVE_VQRDMLASH_qrs8,
      MVE_VQDMLADHXs16,    MVE_VQDMLADHXs32,    MVE_VQDMLADHXs8,
      MVE_VQDMLADHs16,     MVE_VQDMLADHs32,     MVE_VQDMLADHs8,
      MVE_VQDMLSDHXs16,    MVE_VQDMLSDHXs32,    MVE_VQDMLSDHXs8,
      MVE_VQDMLSDHs16,     MVE_VQDMLSDHs32,     MVE_VQDMLSDHs8,
      MVE_VQRDMLADHXs16,   MVE_VQRDMLADHXs32,   MVE_VQRDMLADHXs8,
      MVE_VQRDMLADHs16,    MVE_VQRDMLADHs32,    MVE_VQRDMLADHs8,
      MVE_VQRDMLSDHXs16,   MVE_VQRDMLSDHXs32,   MVE_VQRDMLSDHXs8,
      MVE_VQRDMLSDHs16,    MVE_VQRDMLSDHs32,    MVE_VQRDMLSDHs8,
  };
  for (auto op : isMVEIntMACList) {
    Info[op].IsMVEIntMAC = true;
  }

  std::initializer_list<unsigned> isNonSubwordLoadList = {
      t2LDRi12, t2LDRi8,  t2LDR_POST,  t2LDR_PRE,  t2LDRpci,
      t2LDRs,   t2LDRDi8, t2LDRD_POST, t2LDRD_PRE, tLDRi,
      tLDRpci,  tLDRr,    tLDRspi,
  };
  for (auto op : isNonSubwordLoadList) {
    Info[op].IsNonSubwordLoad = true;
  }

  std::initializer_list<unsigned> isRevList = {
      t2REV, t2REV16, t2REVSH, t2RBIT, tREV, tREV16, tREVSH,
  };
  for (auto op : isRevList) {
    Info[op].IsRev = true;
  }

  std::initializer_list<unsigned> isShiftList = {
      t2ASRri, t2ASRrr, t2LSLri, t2LSLrr, t2LSRri, t2LSRrr, t2RORri, t2RORrr,
      tASRri,  tASRrr,  tLSLSri, tLSLri,  tLSLrr,  tLSRri,  tLSRrr,  tROR,
  };
  for (auto op : isShiftList) {
    Info[op].IsShift = true;
  }

  std::initializer_list<unsigned> Address1List = {
      t2LDRBi12,
      t2LDRBi8,
      t2LDRBpci,
      t2LDRBs,
      t2LDRHi12,
      t2LDRHi8,
      t2LDRHpci,
      t2LDRHs,
      t2LDRSBi12,
      t2LDRSBi8,
      t2LDRSBpci,
      t2LDRSBs,
      t2LDRSHi12,
      t2LDRSHi8,
      t2LDRSHpci,
      t2LDRSHs,
      t2LDRi12,
      t2LDRi8,
      t2LDRpci,
      t2LDRs,
      tLDRBi,
      tLDRBr,
      tLDRHi,
      tLDRHr,
      tLDRSB,
      tLDRSH,
      tLDRi,
      tLDRpci,
      tLDRr,
      tLDRspi,
      t2STRBi12,
      t2STRBi8,
      t2STRBs,
      t2STRHi12,
      t2STRHi8,
      t2STRHs,
      t2STRi12,
      t2STRi8,
      t2STRs,
      tSTRBi,
      tSTRBr,
      tSTRHi,
      tSTRHr,
      tSTRi,
      tSTRr,
      tSTRspi,
      VLDRD,
      VLDRH,
      VLDRS,
      VSTRD,
      VSTRH,
      VSTRS,
      MVE_VLD20_16,
      MVE_VLD20_32,
      MVE_VLD20_8,
      MVE_VLD21_16,
      MVE_VLD21_32,
      MVE_VLD21_8,
      MVE_VLD40_16,
      MVE_VLD40_32,
      MVE_VLD40_8,
      MVE_VLD41_16,
      MVE_VLD41_32,
      MVE_VLD41_8,
      MVE_VLD42_16,
      MVE_VLD42_32,
      MVE_VLD42_8,
      MVE_VLD43_16,
      MVE_VLD43_32,
      MVE_VLD43_8,
      MVE_VLDRBS16,
      MVE_VLDRBS16_rq,
      MVE_VLDRBS32,
      MVE_VLDRBS32_rq,
      MVE_VLDRBU16,
      MVE_VLDRBU16_rq,
      MVE_VLDRBU32,
      MVE_VLDRBU32_rq,
      MVE_VLDRBU8,
      MVE_VLDRBU8_rq,
      MVE_VLDRDU64_qi,
      MVE_VLDRDU64_rq,
      MVE_VLDRDU64_rq_u,
      MVE_VLDRHS32,
      MVE_VLDRHS32_rq,
      MVE_VLDRHS32_rq_u,
      MVE_VLDRHU16,
      MVE_VLDRHU16_rq,
      MVE_VLDRHU16_rq_u,
      MVE_VLDRHU32,
      MVE_VLDRHU32_rq,
      MVE_VLDRHU32_rq_u,
      MVE_VLDRWU32,
      MVE_VLDRWU32_qi,
      MVE_VLDRWU32_rq,
      MVE_VLDRWU32_rq_u,
      MVE_VST20_16,
      MVE_VST20_32,
      MVE_VST20_8,
      MVE_VST21_16,
      MVE_VST21_32,
      MVE_VST21_8,
      MVE_VST40_16,
      MVE_VST40_32,
      MVE_VST40_8,
      MVE_VST41_16,
      MVE_VST41_32,
      MVE_VST41_8,
      MVE_VST42_16,
      MVE_VST42_32,
      MVE_VST42_8,
      MVE_VST43_16,
      MVE_VST43_32,
      MVE_VST43_8,
      MVE_VSTRB16,
      MVE_VSTRB16_rq,
      MVE_VSTRB32,
      MVE_VSTRB32_rq,
      MVE_VSTRBU8,
      MVE_VSTRB8_rq,
      MVE_VSTRD64_qi,
      MVE_VSTRD64_rq,
      MVE_VSTRD64_rq_u,
      MVE_VSTRH32,
      MVE_VSTRH32_rq,
      MVE_VSTRH32_rq_u,
      MVE_VSTRHU16,
      MVE_VSTRH16_rq,
      MVE_VSTRH16_rq_u,
      MVE_VSTRWU32,
      MVE_VSTRW32_qi,
      MVE_VSTRW32_rq,
      MVE_VSTRW32_rq_u,
  };
  std::initializer_list<unsigned> Address2List = {
      t2LDRB_POST,
      t2LDRB_PRE,
      t2LDRDi8,
      t2LDRH_POST,
      t2LDRH_PRE,
      t2LDRSB_POST,
      t2LDRSB_PRE,
      t2LDRSH_POST,
      t2LDRSH_PRE,
      t2LDR_POST,
      t2LDR_PRE,
      t2STRB_POST,
      t2STRB_PRE,
      t2STRDi8,
      t2STRH_POST,
      t2STRH_PRE,
      t2STR_POST,
      t2STR_PRE,
      MVE_VLD20_16_wb,
      MVE_VLD20_32_wb,
      MVE_VLD20_8_wb,
      MVE_VLD21_16_wb,
      MVE_VLD21_32_wb,
      MVE_VLD21_8_wb,
      MVE_VLD40_16_wb,
      MVE_VLD40_32_wb,
      MVE_VLD40_8_wb,
      MVE_VLD41_16_wb,
      MVE_VLD41_32_wb,
      MVE_VLD41_8_wb,
      MVE_VLD42_16_wb,
      MVE_VLD42_32_wb,
      MVE_VLD42_8_wb,
      MVE_VLD43_16_wb,
      MVE_VLD43_32_wb,
      MVE_VLD43_8_wb,
      MVE_VLDRBS16_post,
      MVE_VLDRBS16_pre,
      MVE_VLDRBS32_post,
      MVE_VLDRBS32_pre,
      MVE_VLDRBU16_post,
      MVE_VLDRBU16_pre,
      MVE_VLDRBU32_post,
      MVE_VLDRBU32_pre,
      MVE_VLDRBU8_post,
      MVE_VLDRBU8_pre,
      MVE_VLDRDU64_qi_pre,
      MVE_VLDRHS32_post,
      MVE_VLDRHS32_pre,
      MVE_VLDRHU16_post,
      MVE_VLDRHU16_pre,
      MVE_VLDRHU32_post,
      MVE_VLDRHU32_pre,
      MVE_VLDRWU32_post,
      MVE_VLDRWU32_pre,
      MVE_VLDRWU32_qi_pre,
      MVE_VST20_16_wb,
      MVE_VST20_32_wb,
      MVE_VST20_8_wb,
      MVE_VST21_16_wb,
      MVE_VST21_32_wb,
      MVE_VST21_8_wb,
      MVE_VST40_16_wb,
      MVE_VST40_32_wb,
      MVE_VST40_8_wb,
      MVE_VST41_16_wb,
      MVE_VST41_32_wb,
      MVE_VST41_8_wb,
      MVE_VST42_16_wb,
      MVE_VST42_32_wb,
      MVE_VST42_8_wb,
      MVE_VST43_16_wb,
      MVE_VST43_32_wb,
      MVE_VST43_8_wb,
      MVE_VSTRB16_post,
      MVE_VSTRB16_pre,
      MVE_VSTRB32_post,
      MVE_VSTRB32_pre,
      MVE_VSTRBU8_post,
      MVE_VSTRBU8_pre,
      MVE_VSTRD64_qi_pre,
      MVE_VSTRH32_post,
      MVE_VSTRH32_pre,
      MVE_VSTRHU16_post,
      MVE_VSTRHU16_pre,
      MVE_VSTRWU32_post,
      MVE_VSTRWU32_pre,
      MVE_VSTRW32_qi_pre,
  };
  std::initializer_list<unsigned> Address3List = {
      t2LDRD_POST,
      t2LDRD_PRE,
      t2STRD_POST,
      t2STRD_PRE,
  };
  // Compute a mask of which operands are involved in address computation
  for (auto &op : Address1List) {
    Info[op].AddressOpMask = 0x6;
  }
  for (auto &op : Address2List) {
    Info[op].AddressOpMask = 0xc;
  }
  for (auto &op : Address3List) {
    Info[op].AddressOpMask = 0x18;
  }
  for (auto &op : hasBRegAddrShiftList) {
    Info[op].AddressOpMask |= 0x8;
  }
}

void InstructionInformation::markDPProducersConsumers(
    const ARMBaseInstrInfo *TII) {
  // Learn about all instructions which have FP source/dest registers
  for (unsigned MI = 0; MI < ARM::INSTRUCTION_LIST_END; ++MI) {
    const MCInstrDesc &MID = TII->get(MI);
    auto Operands = MID.operands();
    for (unsigned OI = 0, OIE = MID.getNumOperands(); OI != OIE; ++OI) {
      bool MarkQP = false, MarkDP = false, MarkSP = false;
      switch (Operands[OI].RegClass) {
      case ARM::MQPRRegClassID:
      case ARM::DPRRegClassID:
      case ARM::DPR_8RegClassID:
      case ARM::DPR_VFP2RegClassID:
      case ARM::DPairRegClassID:
      case ARM::DPairSpcRegClassID:
      case ARM::DQuadRegClassID:
      case ARM::DQuadSpcRegClassID:
      case ARM::DTripleRegClassID:
      case ARM::DTripleSpcRegClassID:
        MarkDP = true;
        break;
      case ARM::QPRRegClassID:
      case ARM::QPR_8RegClassID:
      case ARM::QPR_VFP2RegClassID:
      case ARM::QQPRRegClassID:
      case ARM::QQQQPRRegClassID:
        MarkQP = true;
        break;
      case ARM::SPRRegClassID:
      case ARM::SPR_8RegClassID:
      case ARM::FPWithVPRRegClassID:
        MarkSP = true;
        break;
      default:
        break;
      }
      if (MarkQP) {
        if (OI < MID.getNumDefs())
          Info[MI].ProducesQP = true;
        else
          Info[MI].ConsumesQP = true;
      }
      if (MarkDP) {
        if (OI < MID.getNumDefs())
          Info[MI].ProducesDP = true;
        else
          Info[MI].ConsumesDP = true;
      }
      if (MarkSP) {
        if (OI < MID.getNumDefs())
          Info[MI].ProducesSP = true;
        else
          Info[MI].ConsumesSP = true;
      }
    }
  }
}

} // anonymous namespace

static bool hasImplicitCPSRUse(const MachineInstr *MI) {
  return MI->getDesc().hasImplicitUseOfPhysReg(ARM::CPSR);
}

void ARMOverrideBypasses::setBidirLatencies(SUnit &SrcSU, SDep &SrcDep,
                                            unsigned latency) {
  SDep Reverse = SrcDep;
  Reverse.setSUnit(&SrcSU);
  for (SDep &PDep : SrcDep.getSUnit()->Preds) {
    if (PDep == Reverse) {
      PDep.setLatency(latency);
      SrcDep.getSUnit()->setDepthDirty();
      break;
    }
  }
  SrcDep.setLatency(latency);
  SrcSU.setHeightDirty();
}

static bool mismatchedPred(ARMCC::CondCodes a, ARMCC::CondCodes b) {
  return (a & 0xe) != (b & 0xe);
}

// Set output dependences to zero latency for processors which can
// simultaneously issue to the same register.  Returns true if a change
// was made.
bool ARMOverrideBypasses::zeroOutputDependences(SUnit &ISU, SDep &Dep) {
  if (Dep.getKind() == SDep::Output) {
    setBidirLatencies(ISU, Dep, 0);
    return true;
  }
  return false;
}

// The graph doesn't look inside of bundles to determine their
// scheduling boundaries and reports zero latency into and out of them
// (except for CPSR into the bundle, which has latency 1).
// Make some better scheduling assumptions:
// 1) CPSR uses have zero latency; other uses have incoming latency 1
// 2) CPSR defs retain a latency of zero; others have a latency of 1.
//
// Returns 1 if a use change was made; 2 if a def change was made; 0 otherwise
unsigned ARMOverrideBypasses::makeBundleAssumptions(SUnit &ISU, SDep &Dep) {

  SUnit &DepSU = *Dep.getSUnit();
  const MachineInstr *SrcMI = ISU.getInstr();
  unsigned SrcOpcode = SrcMI->getOpcode();
  const MachineInstr *DstMI = DepSU.getInstr();
  unsigned DstOpcode = DstMI->getOpcode();

  if (DstOpcode == ARM::BUNDLE && TII->isPredicated(*DstMI)) {
    setBidirLatencies(
        ISU, Dep,
        (Dep.isAssignedRegDep() && Dep.getReg() == ARM::CPSR) ? 0 : 1);
    return 1;
  }
  if (SrcOpcode == ARM::BUNDLE && TII->isPredicated(*SrcMI) &&
      Dep.isAssignedRegDep() && Dep.getReg() != ARM::CPSR) {
    setBidirLatencies(ISU, Dep, 1);
    return 2;
  }
  return 0;
}

// Determine whether there is a memory RAW hazard here and set up latency
// accordingly
bool ARMOverrideBypasses::memoryRAWHazard(SUnit &ISU, SDep &Dep,
                                          unsigned latency) {
  if (!Dep.isNormalMemory())
    return false;
  auto &SrcInst = *ISU.getInstr();
  auto &DstInst = *Dep.getSUnit()->getInstr();
  if (!SrcInst.mayStore() || !DstInst.mayLoad())
    return false;

  auto SrcMO = *SrcInst.memoperands().begin();
  auto DstMO = *DstInst.memoperands().begin();
  auto SrcVal = SrcMO->getValue();
  auto DstVal = DstMO->getValue();
  auto SrcPseudoVal = SrcMO->getPseudoValue();
  auto DstPseudoVal = DstMO->getPseudoValue();
  if (SrcVal && DstVal && AA->alias(SrcVal, DstVal) == AliasResult::MustAlias &&
      SrcMO->getOffset() == DstMO->getOffset()) {
    setBidirLatencies(ISU, Dep, latency);
    return true;
  } else if (SrcPseudoVal && DstPseudoVal &&
             SrcPseudoVal->kind() == DstPseudoVal->kind() &&
             SrcPseudoVal->kind() == PseudoSourceValue::FixedStack) {
    // Spills/fills
    auto FS0 = cast<FixedStackPseudoSourceValue>(SrcPseudoVal);
    auto FS1 = cast<FixedStackPseudoSourceValue>(DstPseudoVal);
    if (FS0 == FS1) {
      setBidirLatencies(ISU, Dep, latency);
      return true;
    }
  }
  return false;
}

namespace {

std::unique_ptr<InstructionInformation> II;

class CortexM7InstructionInformation : public InstructionInformation {
public:
  CortexM7InstructionInformation(const ARMBaseInstrInfo *TII)
      : InstructionInformation(TII) {}
};

class CortexM7Overrides : public ARMOverrideBypasses {
public:
  CortexM7Overrides(const ARMBaseInstrInfo *TII, AAResults *AA)
      : ARMOverrideBypasses(TII, AA) {
    if (!II)
      II.reset(new CortexM7InstructionInformation(TII));
  }

  void modifyBypasses(SUnit &) override;
};

void CortexM7Overrides::modifyBypasses(SUnit &ISU) {
  const MachineInstr *SrcMI = ISU.getInstr();
  unsigned SrcOpcode = SrcMI->getOpcode();
  bool isNSWload = II->isNonSubwordLoad(SrcOpcode);

  // Walk the successors looking for latency overrides that are needed
  for (SDep &Dep : ISU.Succs) {

    // Output dependences should have 0 latency, as M7 is able to
    // schedule writers to the same register for simultaneous issue.
    if (zeroOutputDependences(ISU, Dep))
      continue;

    if (memoryRAWHazard(ISU, Dep, 4))
      continue;

    // Ignore dependencies other than data
    if (Dep.getKind() != SDep::Data)
      continue;

    SUnit &DepSU = *Dep.getSUnit();
    if (DepSU.isBoundaryNode())
      continue;

    if (makeBundleAssumptions(ISU, Dep) == 1)
      continue;

    const MachineInstr *DstMI = DepSU.getInstr();
    unsigned DstOpcode = DstMI->getOpcode();

    // Word loads into any multiply or divide instruction are considered
    // cannot bypass their scheduling stage. Didn't do this in the .td file
    // because we cannot easily create a read advance that is 0 from certain
    // writer classes and 1 from all the rest.
    // (The other way around would have been easy.)
    if (isNSWload && (II->isMultiply(DstOpcode) || II->isDivide(DstOpcode)))
      setBidirLatencies(ISU, Dep, Dep.getLatency() + 1);

    // Word loads into B operand of a load/store are considered cannot bypass
    // their scheduling stage. Cannot do in the .td file because
    // need to decide between -1 and -2 for ReadAdvance
    if (isNSWload && II->hasBRegAddr(DstOpcode) &&
        DstMI->getOperand(2).getReg() == Dep.getReg())
      setBidirLatencies(ISU, Dep, Dep.getLatency() + 1);

    // Multiplies into any address generation cannot bypass from EX3.  Cannot do
    // in the .td file because need to decide between -1 and -2 for ReadAdvance
    if (II->isMultiply(SrcOpcode)) {
      unsigned OpMask = II->getAddressOpMask(DstOpcode) >> 1;
      for (unsigned i = 1; OpMask; ++i, OpMask >>= 1) {
        if ((OpMask & 1) && DstMI->getOperand(i).isReg() &&
            DstMI->getOperand(i).getReg() == Dep.getReg()) {
          setBidirLatencies(ISU, Dep, 4); // first legal bypass is EX4->EX1
          break;
        }
      }
    }

    // Mismatched conditional producers take longer on M7; they end up looking
    // like they were produced at EX3 and read at IS.
    if (TII->isPredicated(*SrcMI) && Dep.isAssignedRegDep() &&
        (SrcOpcode == ARM::BUNDLE ||
         mismatchedPred(TII->getPredicate(*SrcMI),
                        TII->getPredicate(*DstMI)))) {
      unsigned Lat = 1;
      // Operand A of shift+ALU is treated as an EX1 read instead of EX2.
      if (II->isInlineShiftALU(DstOpcode) && DstMI->getOperand(3).getImm() &&
          DstMI->getOperand(1).getReg() == Dep.getReg())
        Lat = 2;
      Lat = std::min(3u, Dep.getLatency() + Lat);
      setBidirLatencies(ISU, Dep, std::max(Dep.getLatency(), Lat));
    }

    // CC setter into conditional producer shouldn't have a latency of more
    // than 1 unless it's due to an implicit read. (All the "true" readers
    // of the condition code use an implicit read, and predicates use an
    // explicit.)
    if (Dep.isAssignedRegDep() && Dep.getReg() == ARM::CPSR &&
        TII->isPredicated(*DstMI) && !hasImplicitCPSRUse(DstMI))
      setBidirLatencies(ISU, Dep, 1);

    // REV instructions cannot bypass directly into the EX1 shifter.  The
    // code is slightly inexact as it doesn't attempt to ensure that the bypass
    // is to the shifter operands.
    if (II->isRev(SrcOpcode)) {
      if (II->isInlineShiftALU(DstOpcode))
        setBidirLatencies(ISU, Dep, 2);
      else if (II->isShift(DstOpcode))
        setBidirLatencies(ISU, Dep, 1);
    }
  }
}

class M85InstructionInformation : public InstructionInformation {
public:
  M85InstructionInformation(const ARMBaseInstrInfo *t)
      : InstructionInformation(t) {
    markDPProducersConsumers(t);
  }
};

class M85Overrides : public ARMOverrideBypasses {
public:
  M85Overrides(const ARMBaseInstrInfo *t, AAResults *a)
      : ARMOverrideBypasses(t, a) {
    if (!II)
      II.reset(new M85InstructionInformation(t));
  }

  void modifyBypasses(SUnit &) override;

private:
  unsigned computeBypassStage(const MCSchedClassDesc *SCD);
  signed modifyMixedWidthFP(const MachineInstr *SrcMI,
                            const MachineInstr *DstMI, unsigned RegID,
                            const MCSchedClassDesc *SCD);
};

unsigned M85Overrides::computeBypassStage(const MCSchedClassDesc *SCDesc) {
  auto SM = DAG->getSchedModel();
  unsigned DefIdx = 0; // just look for the first output's timing
  if (DefIdx < SCDesc->NumWriteLatencyEntries) {
    // Lookup the definition's write latency in SubtargetInfo.
    const MCWriteLatencyEntry *WLEntry =
        SM->getSubtargetInfo()->getWriteLatencyEntry(SCDesc, DefIdx);
    unsigned Latency = WLEntry->Cycles >= 0 ? WLEntry->Cycles : 1000;
    if (Latency == 4)
      return 2;
    else if (Latency == 5)
      return 3;
    else if (Latency > 3)
      return 3;
    else
      return Latency;
  }
  return 2;
}

// Latency changes for bypassing between FP registers of different sizes:
//
// Note that mixed DP/SP are unlikely because of the semantics
// of C.  Mixed MVE/SP are quite common when MVE intrinsics are used.
signed M85Overrides::modifyMixedWidthFP(const MachineInstr *SrcMI,
                                        const MachineInstr *DstMI,
                                        unsigned RegID,
                                        const MCSchedClassDesc *SCD) {

  if (!II->producesSP(SrcMI->getOpcode()) &&
      !II->producesDP(SrcMI->getOpcode()) &&
      !II->producesQP(SrcMI->getOpcode()))
    return 0;

  if (Register::isVirtualRegister(RegID)) {
    if (II->producesSP(SrcMI->getOpcode()) &&
        II->consumesDP(DstMI->getOpcode())) {
      for (auto &OP : SrcMI->operands())
        if (OP.isReg() && OP.isDef() && OP.getReg() == RegID &&
            OP.getSubReg() == ARM::ssub_1)
          return 5 - computeBypassStage(SCD);
    } else if (II->producesSP(SrcMI->getOpcode()) &&
               II->consumesQP(DstMI->getOpcode())) {
      for (auto &OP : SrcMI->operands())
        if (OP.isReg() && OP.isDef() && OP.getReg() == RegID &&
            (OP.getSubReg() == ARM::ssub_1 || OP.getSubReg() == ARM::ssub_3))
          return 5 - computeBypassStage(SCD) -
                 ((OP.getSubReg() == ARM::ssub_2 ||
                   OP.getSubReg() == ARM::ssub_3)
                      ? 1
                      : 0);
    } else if (II->producesDP(SrcMI->getOpcode()) &&
               II->consumesQP(DstMI->getOpcode())) {
      for (auto &OP : SrcMI->operands())
        if (OP.isReg() && OP.isDef() && OP.getReg() == RegID &&
            OP.getSubReg() == ARM::ssub_1)
          return -1;
    } else if (II->producesDP(SrcMI->getOpcode()) &&
               II->consumesSP(DstMI->getOpcode())) {
      for (auto &OP : DstMI->operands())
        if (OP.isReg() && OP.isUse() && OP.getReg() == RegID &&
            OP.getSubReg() == ARM::ssub_1)
          return 5 - computeBypassStage(SCD);
    } else if (II->producesQP(SrcMI->getOpcode()) &&
               II->consumesSP(DstMI->getOpcode())) {
      for (auto &OP : DstMI->operands())
        if (OP.isReg() && OP.isUse() && OP.getReg() == RegID &&
            (OP.getSubReg() == ARM::ssub_1 || OP.getSubReg() == ARM::ssub_3))
          return 5 - computeBypassStage(SCD) +
                 ((OP.getSubReg() == ARM::ssub_2 ||
                   OP.getSubReg() == ARM::ssub_3)
                      ? 1
                      : 0);
    } else if (II->producesQP(SrcMI->getOpcode()) &&
               II->consumesDP(DstMI->getOpcode())) {
      for (auto &OP : DstMI->operands())
        if (OP.isReg() && OP.isUse() && OP.getReg() == RegID &&
            OP.getSubReg() == ARM::ssub_1)
          return 1;
    }
  } else if (Register::isPhysicalRegister(RegID)) {
    // Note that when the producer is narrower, not all of the producers
    // may be present in the scheduling graph; somewhere earlier in the
    // compiler, an implicit def/use of the aliased full register gets
    // added to the producer, and so only that producer is seen as *the*
    // single producer.  This behavior also has the unfortunate effect of
    // serializing the producers in the compiler's view of things.
    if (II->producesSP(SrcMI->getOpcode()) &&
        II->consumesDP(DstMI->getOpcode())) {
      for (auto &OP : SrcMI->operands())
        if (OP.isReg() && OP.isDef() && OP.getReg() >= ARM::S1 &&
            OP.getReg() <= ARM::S31 && (OP.getReg() - ARM::S0) % 2 &&
            (OP.getReg() == RegID ||
             (OP.getReg() - ARM::S0) / 2 + ARM::D0 == RegID ||
             (OP.getReg() - ARM::S0) / 4 + ARM::Q0 == RegID))
          return 5 - computeBypassStage(SCD);
    } else if (II->producesSP(SrcMI->getOpcode()) &&
               II->consumesQP(DstMI->getOpcode())) {
      for (auto &OP : SrcMI->operands())
        if (OP.isReg() && OP.isDef() && OP.getReg() >= ARM::S1 &&
            OP.getReg() <= ARM::S31 && (OP.getReg() - ARM::S0) % 2 &&
            (OP.getReg() == RegID ||
             (OP.getReg() - ARM::S0) / 2 + ARM::D0 == RegID ||
             (OP.getReg() - ARM::S0) / 4 + ARM::Q0 == RegID))
          return 5 - computeBypassStage(SCD) -
                 (((OP.getReg() - ARM::S0) / 2) % 2 ? 1 : 0);
    } else if (II->producesDP(SrcMI->getOpcode()) &&
               II->consumesQP(DstMI->getOpcode())) {
      for (auto &OP : SrcMI->operands())
        if (OP.isReg() && OP.isDef() && OP.getReg() >= ARM::D0 &&
            OP.getReg() <= ARM::D15 && (OP.getReg() - ARM::D0) % 2 &&
            (OP.getReg() == RegID ||
             (OP.getReg() - ARM::D0) / 2 + ARM::Q0 == RegID))
          return -1;
    } else if (II->producesDP(SrcMI->getOpcode()) &&
               II->consumesSP(DstMI->getOpcode())) {
      if (RegID >= ARM::S1 && RegID <= ARM::S31 && (RegID - ARM::S0) % 2)
        return 5 - computeBypassStage(SCD);
    } else if (II->producesQP(SrcMI->getOpcode()) &&
               II->consumesSP(DstMI->getOpcode())) {
      if (RegID >= ARM::S1 && RegID <= ARM::S31 && (RegID - ARM::S0) % 2)
        return 5 - computeBypassStage(SCD) +
               (((RegID - ARM::S0) / 2) % 2 ? 1 : 0);
    } else if (II->producesQP(SrcMI->getOpcode()) &&
               II->consumesDP(DstMI->getOpcode())) {
      if (RegID >= ARM::D1 && RegID <= ARM::D15 && (RegID - ARM::D0) % 2)
        return 1;
    }
  }
  return 0;
}

void M85Overrides::modifyBypasses(SUnit &ISU) {
  const MachineInstr *SrcMI = ISU.getInstr();
  unsigned SrcOpcode = SrcMI->getOpcode();
  bool isNSWload = II->isNonSubwordLoad(SrcOpcode);

  // Walk the successors looking for latency overrides that are needed
  for (SDep &Dep : ISU.Succs) {

    // Output dependences should have 0 latency, as CortexM85 is able to
    // schedule writers to the same register for simultaneous issue.
    if (zeroOutputDependences(ISU, Dep))
      continue;

    if (memoryRAWHazard(ISU, Dep, 3))
      continue;

    // Ignore dependencies other than data or strong ordering.
    if (Dep.getKind() != SDep::Data)
      continue;

    SUnit &DepSU = *Dep.getSUnit();
    if (DepSU.isBoundaryNode())
      continue;

    if (makeBundleAssumptions(ISU, Dep) == 1)
      continue;

    const MachineInstr *DstMI = DepSU.getInstr();
    unsigned DstOpcode = DstMI->getOpcode();

    // Word loads into B operand of a load/store with cannot bypass their
    // scheduling stage. Cannot do in the .td file because need to decide
    // between -1 and -2 for ReadAdvance

    if (isNSWload && II->hasBRegAddrShift(DstOpcode) &&
        DstMI->getOperand(3).getImm() != 0 && // shift operand
        DstMI->getOperand(2).getReg() == Dep.getReg())
      setBidirLatencies(ISU, Dep, Dep.getLatency() + 1);

    if (isNSWload && isMVEVectorInstruction(DstMI)) {
      setBidirLatencies(ISU, Dep, Dep.getLatency() + 1);
    }

    if (II->isMVEIntMAC(DstOpcode) &&
        II->isMVEIntMACMatched(SrcOpcode, DstOpcode) &&
        DstMI->getOperand(0).isReg() &&
        DstMI->getOperand(0).getReg() == Dep.getReg())
      setBidirLatencies(ISU, Dep, Dep.getLatency() - 1);

    // CC setter into conditional producer shouldn't have a latency of more
    // than 0 unless it's due to an implicit read.
    if (Dep.isAssignedRegDep() && Dep.getReg() == ARM::CPSR &&
        TII->isPredicated(*DstMI) && !hasImplicitCPSRUse(DstMI))
      setBidirLatencies(ISU, Dep, 0);

    if (signed ALat = modifyMixedWidthFP(SrcMI, DstMI, Dep.getReg(),
                                         DAG->getSchedClass(&ISU)))
      setBidirLatencies(ISU, Dep, std::max(0, signed(Dep.getLatency()) + ALat));

    if (II->isRev(SrcOpcode)) {
      if (II->isInlineShiftALU(DstOpcode))
        setBidirLatencies(ISU, Dep, 1);
      else if (II->isShift(DstOpcode))
        setBidirLatencies(ISU, Dep, 1);
    }
  }
}

// Add M55 specific overrides for latencies between instructions. Currently it:
//  - Adds an extra cycle latency between MVE VMLAV and scalar instructions.
class CortexM55Overrides : public ARMOverrideBypasses {
public:
  CortexM55Overrides(const ARMBaseInstrInfo *TII, AAResults *AA)
      : ARMOverrideBypasses(TII, AA) {}

  void modifyBypasses(SUnit &SU) override {
    MachineInstr *SrcMI = SU.getInstr();
    if (!(SrcMI->getDesc().TSFlags & ARMII::HorizontalReduction))
      return;

    for (SDep &Dep : SU.Succs) {
      if (Dep.getKind() != SDep::Data)
        continue;
      SUnit &DepSU = *Dep.getSUnit();
      if (DepSU.isBoundaryNode())
        continue;
      MachineInstr *DstMI = DepSU.getInstr();

      if (!isMVEVectorInstruction(DstMI) && !DstMI->mayStore())
        setBidirLatencies(SU, Dep, 3);
    }
  }
};

} // end anonymous namespace

void ARMOverrideBypasses::apply(ScheduleDAGInstrs *DAGInstrs) {
  DAG = DAGInstrs;
  for (SUnit &ISU : DAGInstrs->SUnits) {
    if (ISU.isBoundaryNode())
      continue;
    modifyBypasses(ISU);
  }
  if (DAGInstrs->ExitSU.getInstr())
    modifyBypasses(DAGInstrs->ExitSU);
}

std::unique_ptr<ScheduleDAGMutation>
createARMLatencyMutations(const ARMSubtarget &ST, AAResults *AA) {
  if (ST.isCortexM85())
    return std::make_unique<M85Overrides>(ST.getInstrInfo(), AA);
  else if (ST.isCortexM7())
    return std::make_unique<CortexM7Overrides>(ST.getInstrInfo(), AA);
  else if (ST.isCortexM55())
    return std::make_unique<CortexM55Overrides>(ST.getInstrInfo(), AA);

  return nullptr;
}

} // end namespace llvm
