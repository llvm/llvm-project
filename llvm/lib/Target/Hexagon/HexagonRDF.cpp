//===--- HexagonRDF.cpp ---------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "HexagonRDF.h"
#include "HexagonInstrInfo.h"
#include "HexagonRegisterInfo.h"

#include "llvm/CodeGen/MachineInstr.h"

using namespace llvm;
using namespace rdf;

bool HexagonRegisterAliasInfo::covers(RegisterRef RA, RegisterRef RB,
      const DataFlowGraph &DFG) const {
  if (RA == RB)
    return true;

  if (TargetRegisterInfo::isVirtualRegister(RA.Reg) &&
      TargetRegisterInfo::isVirtualRegister(RB.Reg)) {
    // Hexagon-specific cases.
    if (RA.Reg == RB.Reg) {
      if (RA.Sub == 0)
        return true;
      if (RB.Sub == 0)
        return false;
    }
  }

  return RegisterAliasInfo::covers(RA, RB, DFG);
}

bool HexagonRegisterAliasInfo::covers(const RegisterSet &RRs, RegisterRef RR,
      const DataFlowGraph &DFG) const {
  if (RRs.count(RR))
    return true;

  // The exact reference RR is not in the set.

  if (TargetRegisterInfo::isVirtualRegister(RR.Reg)) {
    // Check if the there are references in RRs of the same register,
    // with both covering subregisters.
    bool HasLo = RRs.count({RR.Reg, Hexagon::subreg_loreg});
    bool HasHi = RRs.count({RR.Reg, Hexagon::subreg_hireg});
    if (HasLo && HasHi)
      return true;
  }

  if (TargetRegisterInfo::isPhysicalRegister(RR.Reg)) {
    // Check if both covering subregisters are present with full
    // lane masks.
    unsigned Lo = TRI.getSubReg(RR.Reg, Hexagon::subreg_loreg);
    unsigned Hi = TRI.getSubReg(RR.Reg, Hexagon::subreg_hireg);
    if (RRs.count({Lo, 0}) && RRs.count({Hi, 0}))
      return true;
  }

  return RegisterAliasInfo::covers(RRs, RR, DFG);
}
