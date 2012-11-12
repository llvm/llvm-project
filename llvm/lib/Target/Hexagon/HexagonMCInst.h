//===- HexagonMCInst.h - Hexagon sub-class of MCInst ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class extends MCInst to allow some VLIW annotation.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGONMCINST_H
#define HEXAGONMCINST_H

#include "llvm/MC/MCInst.h"
#include "llvm/CodeGen/MachineInstr.h"

namespace llvm {
  class HexagonMCInst: public MCInst {
    // Packet start and end markers
    unsigned startPacket: 1, endPacket: 1;
    const MachineInstr *MachineI;
  public:
    explicit HexagonMCInst(): MCInst(),
                              startPacket(0), endPacket(0) {}

    const MachineInstr* getMI() const { return MachineI; }

    void setMI(const MachineInstr *MI) { MachineI = MI; }

    bool isStartPacket() const { return (startPacket); }
    bool isEndPacket() const { return (endPacket); }

    void setStartPacket(bool yes) { startPacket = yes; }
    void setEndPacket(bool yes) { endPacket = yes; }
  };
}

#endif
