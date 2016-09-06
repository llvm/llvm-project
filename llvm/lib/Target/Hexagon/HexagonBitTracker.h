//===--- HexagonBitTracker.h ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGONBITTRACKER_H
#define HEXAGONBITTRACKER_H

#include "BitTracker.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {
  class HexagonInstrInfo;
  class HexagonRegisterInfo;

struct HexagonEvaluator : public BitTracker::MachineEvaluator {
  typedef BitTracker::CellMapType CellMapType;
  typedef BitTracker::RegisterRef RegisterRef;
  typedef BitTracker::RegisterCell RegisterCell;
  typedef BitTracker::BranchTargetList BranchTargetList;

  HexagonEvaluator(const HexagonRegisterInfo &tri, MachineRegisterInfo &mri,
                   const HexagonInstrInfo &tii, MachineFunction &mf);

  bool evaluate(const MachineInstr &MI, const CellMapType &Inputs,
                CellMapType &Outputs) const override;
  bool evaluate(const MachineInstr &BI, const CellMapType &Inputs,
                BranchTargetList &Targets, bool &FallsThru) const override;

  BitTracker::BitMask mask(unsigned Reg, unsigned Sub) const override;

  MachineFunction &MF;
  MachineFrameInfo &MFI;
  const HexagonInstrInfo &TII;

private:
  bool evaluateLoad(const MachineInstr &MI, const CellMapType &Inputs,
                    CellMapType &Outputs) const;
  bool evaluateFormalCopy(const MachineInstr &MI, const CellMapType &Inputs,
                          CellMapType &Outputs) const;

  unsigned getNextPhysReg(unsigned PReg, unsigned Width) const;
  unsigned getVirtRegFor(unsigned PReg) const;

  // Type of formal parameter extension.
  struct ExtType {
    enum { SExt, ZExt };
    char Type;
    uint16_t Width;
    ExtType() : Type(0), Width(0) {}
    ExtType(char t, uint16_t w) : Type(t), Width(w) {}
  };
  // Map VR -> extension type.
  typedef DenseMap<unsigned, ExtType> RegExtMap;
  RegExtMap VRX;
};

} // end namespace llvm

#endif
