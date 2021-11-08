//===- PPCMacroFusion.cpp - PowerPC Macro Fusion --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains the PowerPC implementation of the DAG scheduling
///  mutation to pair instructions back to back.
//
//===----------------------------------------------------------------------===//

#include "PPC.h"
#include "PPCSubtarget.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/MacroFusion.h"

using namespace llvm;
namespace {

class FusionFeature {
public:
  typedef SmallDenseSet<unsigned> FusionOpSet;

  enum FusionKind {
  #define FUSION_KIND(KIND) FK_##KIND
  #define FUSION_FEATURE(KIND, HAS_FEATURE, DEP_OP_IDX, OPSET1, OPSET2) \
    FUSION_KIND(KIND),
  #include "PPCMacroFusion.def"
  FUSION_KIND(END)
  };
private:
  // Each fusion feature is assigned with one fusion kind. All the
  // instructions with the same fusion kind have the same fusion characteristic.
  FusionKind Kd;
  // True if this feature is enabled.
  bool Supported;
  // li rx, si
  // load rt, ra, rx
  // The dependent operand index in the second op(load). And the negative means
  // it could be any one. 
  int DepOpIdx;
  // The first fusion op set.
  FusionOpSet OpSet1;
  // The second fusion op set.
  FusionOpSet OpSet2;
public:
  FusionFeature(FusionKind Kind, bool HasFeature, int Index,
                const FusionOpSet &First, const FusionOpSet &Second) :
    Kd(Kind), Supported(HasFeature), DepOpIdx(Index), OpSet1(First), 
    OpSet2(Second) {}

  bool hasOp1(unsigned Opc) const { return OpSet1.contains(Opc); }
  bool hasOp2(unsigned Opc) const { return OpSet2.contains(Opc); }
  bool isSupported() const { return Supported; }
  Optional<unsigned> depOpIdx() const {
    if (DepOpIdx < 0)
      return None;
    return DepOpIdx;
  }

  FusionKind getKind() const { return Kd; }
};

static bool matchingRegOps(const MachineInstr &FirstMI,
                           int FirstMIOpIndex,
                           const MachineInstr &SecondMI,
                           int SecondMIOpIndex) {
  const MachineOperand &Op1 = FirstMI.getOperand(FirstMIOpIndex);
  const MachineOperand &Op2 = SecondMI.getOperand(SecondMIOpIndex);
  if (!Op1.isReg() || !Op2.isReg())
    return false;

  return Op1.getReg() == Op2.getReg();
}

static bool matchingImmOps(const MachineInstr &MI,
                           int MIOpIndex,
                           int64_t Expect,
                           unsigned ExtendFrom = 64) {
  const MachineOperand &Op = MI.getOperand(MIOpIndex);
  if (!Op.isImm())
    return false;
  int64_t Imm = Op.getImm();
  if (ExtendFrom < 64)
    Imm = SignExtend64(Imm, ExtendFrom);
  return Imm == Expect;
}

// Return true if the FirstMI meets the constraints of SecondMI according to
// fusion specification.
static bool checkOpConstraints(FusionFeature::FusionKind Kd,
                               const MachineInstr &FirstMI,
                               const MachineInstr &SecondMI) {
  switch (Kd) {
  // The hardware didn't require any specific check for the fused instructions'
  // operands. Therefore, return true to indicate that, it is fusable.
  default: return true;
  // [addi rt,ra,si - lxvd2x xt,ra,rb] etc.
  case FusionFeature::FK_AddiLoad: {
    // lxvd2x(ra) cannot be zero
    const MachineOperand &RA = SecondMI.getOperand(1);
    if (!RA.isReg())
      return true;

    return Register::isVirtualRegister(RA.getReg()) ||
      (RA.getReg() != PPC::ZERO && RA.getReg() != PPC::ZERO8);
  }
  // [addis rt,ra,si - ld rt,ds(ra)] etc.
  case FusionFeature::FK_AddisLoad: {
    const MachineOperand &RT = SecondMI.getOperand(0);
    if (!RT.isReg())
      return true;

    // Only check it for non-virtual register.
    if (!Register::isVirtualRegister(RT.getReg()))
      // addis(rt) = ld(ra) = ld(rt)
      // ld(rt) cannot be zero
      if (!matchingRegOps(SecondMI, 0, SecondMI, 2) ||
          (RT.getReg() == PPC::ZERO || RT.getReg() == PPC::ZERO8))
          return false;

    // addis(si) first 12 bits must be all 1s or all 0s
    const MachineOperand &SI = FirstMI.getOperand(2);
    if (!SI.isImm())
      return true;
    int64_t Imm = SI.getImm();
    if (((Imm & 0xFFF0) != 0) && ((Imm & 0xFFF0) != 0xFFF0))
      return false;

    // If si = 1111111111110000 and the msb of the d/ds field of the load equals
    // 1, then fusion does not occur.
    if ((Imm & 0xFFF0) == 0xFFF0) {
      const MachineOperand &D = SecondMI.getOperand(1);
      if (!D.isImm())
        return true;

      // 14 bit for DS field, while 16 bit for D field.
      int MSB = 15;
      if (SecondMI.getOpcode() == PPC::LD)
        MSB = 13;

      return (D.getImm() & (1ULL << MSB)) == 0;
    }
    return true;
  }

  case FusionFeature::FK_SldiAdd:
    return (matchingImmOps(FirstMI, 2, 3) && matchingImmOps(FirstMI, 3, 60)) ||
           (matchingImmOps(FirstMI, 2, 6) && matchingImmOps(FirstMI, 3, 57));
  }

  llvm_unreachable("All the cases should have been handled");
  return true;
}

/// Check if the instr pair, FirstMI and SecondMI, should be fused together.
/// Given SecondMI, when FirstMI is unspecified, then check if SecondMI may be
/// part of a fused pair at all.
static bool shouldScheduleAdjacent(const TargetInstrInfo &TII,
                                   const TargetSubtargetInfo &TSI,
                                   const MachineInstr *FirstMI,
                                   const MachineInstr &SecondMI) {
  // We use the PPC namespace to avoid the need to prefix opcodes with PPC:: in
  // the def file.
  using namespace PPC;

  const PPCSubtarget &ST = static_cast<const PPCSubtarget&>(TSI);
  static const FusionFeature FusionFeatures[] = {
  #define FUSION_FEATURE(KIND, HAS_FEATURE, DEP_OP_IDX, OPSET1, OPSET2) { \
    FusionFeature::FUSION_KIND(KIND), ST.HAS_FEATURE(), DEP_OP_IDX, { OPSET1 },\
    { OPSET2 } },
   #include "PPCMacroFusion.def"
  };
  #undef FUSION_KIND

  for (auto &Feature : FusionFeatures) {
    // Skip if the feature is not supported.
    if (!Feature.isSupported())
      continue;

    // Only when the SecondMI is fusable, we are starting to look for the
    // fusable FirstMI.
    if (Feature.hasOp2(SecondMI.getOpcode())) {
      // If FirstMI == nullptr, that means, we're only checking whether SecondMI
      // can be fused at all.
      if (!FirstMI)
        return true;

      // Checking if the FirstMI is fusable with the SecondMI.
      if (!Feature.hasOp1(FirstMI->getOpcode()))
        continue;

      auto DepOpIdx = Feature.depOpIdx();
      if (DepOpIdx.hasValue()) {
        // Checking if the result of the FirstMI is the desired operand of the
        // SecondMI if the DepOpIdx is set. Otherwise, ignore it.
        if (!matchingRegOps(*FirstMI, 0, SecondMI, *DepOpIdx))
          return false;
      }
  
      // Checking more on the instruction operands.
      if (checkOpConstraints(Feature.getKind(), *FirstMI, SecondMI))
        return true;
    }
  }

  return false;
}

} // end anonymous namespace

namespace llvm {

std::unique_ptr<ScheduleDAGMutation> createPowerPCMacroFusionDAGMutation () {
  return createMacroFusionDAGMutation(shouldScheduleAdjacent);
}

} // end namespace llvm
