//===- AMDGPUGlobalISelUtils -------------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUGLOBALISELUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUGLOBALISELUTILS_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include <utility>

namespace llvm {

class MachineRegisterInfo;
class GCNSubtarget;
class GISelValueTracking;
class LLT;
class MachineFunction;
class MachineIRBuilder;
class RegisterBankInfo;

namespace AMDGPU {

/// Returns base register and constant offset.
std::pair<Register, unsigned>
getBaseWithConstantOffset(MachineRegisterInfo &MRI, Register Reg,
                          GISelValueTracking *ValueTracking = nullptr,
                          bool CheckNUW = false);

// Currently finds S32/S64 lane masks that can be declared as divergent by
// uniformity analysis (all are phis at the moment).
// These are defined as i32/i64 in some IR intrinsics (not as i1).
// Tablegen forces(via telling that lane mask IR intrinsics are uniform) most of
// S32/S64 lane masks to be uniform, as this results in them ending up with sgpr
// reg class after instruction-select, don't search for all of them.
class IntrinsicLaneMaskAnalyzer {
  SmallDenseSet<Register, 8> S32S64LaneMask;
  MachineRegisterInfo &MRI;

public:
  IntrinsicLaneMaskAnalyzer(MachineFunction &MF);
  bool isS32S64LaneMask(Register Reg) const;

private:
  void initLaneMaskIntrinsics(MachineFunction &MF);
};

void buildReadAnyLane(MachineIRBuilder &B, Register SgprDst, Register VgprSrc,
                      const RegisterBankInfo &RBI);

template <typename T> struct BitOp3Helper {
  BitOp3Helper() = delete;
  BitOp3Helper(const BitOp3Helper &) = delete;
  BitOp3Helper &operator=(const BitOp3Helper &) = delete;
};

template <> struct BitOp3Helper<Register> {
  BitOp3Helper(MachineRegisterInfo *MRI) : MRI(MRI) {}
  bool isAllOnes(Register R) const {
    return mi_match(R, *MRI, MIPatternMatch::m_AllOnesInt());
  }
  bool isZero(Register R) const {
    return mi_match(R, *MRI, MIPatternMatch::m_ZeroInt());
  }
  bool isNot(Register R, Register &LHS) const {
    if (mi_match(R, *MRI, m_Not(MIPatternMatch::m_Reg(LHS)))) {
      LHS = getSrcRegIgnoringCopies(LHS, *MRI);
      return true;
    }
    return false;
  }
  std::pair<Register, Register> getLHSRHS(Register R) {
    MachineInstr *MI = MRI->getVRegDef(R);
    auto LHS = getSrcRegIgnoringCopies(MI->getOperand(1).getReg(), *MRI);
    auto RHS = getSrcRegIgnoringCopies(MI->getOperand(2).getReg(), *MRI);
    return std::make_pair(LHS, RHS);
  }
  unsigned getOpcode(Register R) {
    MachineInstr *MI = MRI->getVRegDef(R);
    switch (MI->getOpcode()) {
    case TargetOpcode::G_AND:
      return ISD::AND;
    case TargetOpcode::G_OR:
      return ISD::OR;
    case TargetOpcode::G_XOR:
      return ISD::XOR;
    default:
      // Use DELETED_NODE as a notion of an unsupported value.
      return ISD::DELETED_NODE;
    }
  }

  MachineRegisterInfo *MRI;
};

template <> struct BitOp3Helper<SDValue> {
  BitOp3Helper(const MachineRegisterInfo *MRI = nullptr) : MRI(MRI) {}
  bool isAllOnes(SDValue Op) const {
    if (auto *C = dyn_cast<ConstantSDNode>(Op))
      if (C->isAllOnes())
        return true;
    return false;
  }
  bool isZero(SDValue Op) const {
    if (auto *C = dyn_cast<ConstantSDNode>(Op))
      if (C->isZero())
        return true;
    return false;
  }
  bool isNot(SDValue Op, SDValue &LHS) const {
    if (Op.getOpcode() == ISD::XOR)
      if (auto *C = dyn_cast<ConstantSDNode>(Op.getOperand(1)))
        if (C->isAllOnes()) {
          LHS = Op.getOperand(0);
          return true;
        }
    return false;
  }
  std::pair<SDValue, SDValue> getLHSRHS(SDValue In) {
    auto LHS = In.getOperand(0);
    auto RHS = In.getOperand(1);
    return std::make_pair(LHS, RHS);
  }
  unsigned getOpcode(SDValue Op) {
    switch (Op.getOpcode()) {
    case ISD::AND:
    case ISD::OR:
    case ISD::XOR:
      return Op.getOpcode();
    default:
      // Use DELETED_NODE as a notion of an unsupported value.
      return ISD::DELETED_NODE;
    }
  }

  [[maybe_unused]] const MachineRegisterInfo *MRI;
};

// Match BITOP3 operation and return a number of matched instructions plus
// truth table.
template <typename T>
static std::pair<unsigned, uint8_t>
BitOp3_Op(llvm::AMDGPU::BitOp3Helper<T> &Helper, T In,
          SmallVectorImpl<T> &Src) {
  unsigned NumOpcodes = 0;
  uint8_t LHSBits, RHSBits;

  auto getOperandBits = [&Helper, &In, &Src](T Op, uint8_t &Bits) -> bool {
    // Define truth table given Src0, Src1, Src2 bits permutations:
    //                          0     0     0
    //                          0     0     1
    //                          0     1     0
    //                          0     1     1
    //                          1     0     0
    //                          1     0     1
    //                          1     1     0
    //                          1     1     1
    const uint8_t SrcBits[3] = {0xf0, 0xcc, 0xaa};

    if (Helper.isAllOnes(Op)) {
      Bits = 0xff;
      return true;
    }
    if (Helper.isZero(Op)) {
      Bits = 0;
      return true;
    }

    for (unsigned I = 0; I < Src.size(); ++I) {
      // Try to find existing reused operand
      if (Src[I] == Op) {
        Bits = SrcBits[I];
        return true;
      }
      // Try to replace parent operator
      if (Src[I] == In) {
        Bits = SrcBits[I];
        Src[I] = Op;
        return true;
      }
    }

    if (Src.size() == 3) {
      // No room left for operands. Try one last time, there can be a 'not' of
      // one of our source operands. In this case we can compute the bits
      // without growing Src vector.
      T LHS;
      if (Helper.isNot(Op, LHS)) {
        for (unsigned I = 0; I < Src.size(); ++I) {
          if (Src[I] == LHS) {
            Bits = ~SrcBits[I];
            return true;
          }
        }
      }

      return false;
    }

    Bits = SrcBits[Src.size()];
    Src.push_back(Op);
    return true;
  };

  switch (Helper.getOpcode(In)) {
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR: {
    auto LHSRHS = Helper.getLHSRHS(In);
    T LHS = std::get<0>(LHSRHS);
    T RHS = std::get<1>(LHSRHS);

    SmallVector<T, 3> Backup(Src.begin(), Src.end());
    if (!getOperandBits(LHS, LHSBits) || !getOperandBits(RHS, RHSBits)) {
      Src = Backup;
      return std::make_pair(0, 0);
    }

    // Recursion is naturally limited by the size of the operand vector.
    auto LHSHelper = BitOp3Helper<decltype(LHS)>(Helper.MRI);
    auto Op = BitOp3_Op(LHSHelper, LHS, Src);
    if (Op.first) {
      NumOpcodes += Op.first;
      LHSBits = Op.second;
    }

    auto RHSHelper = BitOp3Helper<decltype(RHS)>(Helper.MRI);
    Op = BitOp3_Op(RHSHelper, RHS, Src);
    if (Op.first) {
      NumOpcodes += Op.first;
      RHSBits = Op.second;
    }
    break;
  }
  default:
    return std::make_pair(0, 0);
  }

  uint8_t TTbl;
  switch (Helper.getOpcode(In)) {
  case ISD::AND:
    TTbl = LHSBits & RHSBits;
    break;
  case ISD::OR:
    TTbl = LHSBits | RHSBits;
    break;
  case ISD::XOR:
    TTbl = LHSBits ^ RHSBits;
    break;
  default:
    llvm_unreachable("Unhandled opcode");
    break;
  }

  return std::make_pair(NumOpcodes + 1, TTbl);
}

} // namespace AMDGPU
} // namespace llvm

#endif
