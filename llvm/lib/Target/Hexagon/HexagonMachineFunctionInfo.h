//=- HexagonMachineFuctionInfo.h - Hexagon machine function info --*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef HexagonMACHINEFUNCTIONINFO_H
#define HexagonMACHINEFUNCTIONINFO_H

#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

  namespace Hexagon {
    const unsigned int StartPacket = 0x1;
    const unsigned int EndPacket = 0x2;
  }


/// Hexagon target-specific information for each MachineFunction.
class HexagonMachineFunctionInfo : public MachineFunctionInfo {
  // SRetReturnReg - Some subtargets require that sret lowering includes
  // returning the value of the returned struct in a register. This field
  // holds the virtual register into which the sret argument is passed.
  unsigned SRetReturnReg;
  std::vector<MachineInstr*> AllocaAdjustInsts;
  int VarArgsFrameIndex;
  bool HasClobberLR;

  std::map<const MachineInstr*, unsigned> PacketInfo;


public:
  HexagonMachineFunctionInfo() : SRetReturnReg(0), HasClobberLR(0) {}

  HexagonMachineFunctionInfo(MachineFunction &MF) : SRetReturnReg(0),
                                                    HasClobberLR(0) {}

  unsigned getSRetReturnReg() const { return SRetReturnReg; }
  void setSRetReturnReg(unsigned Reg) { SRetReturnReg = Reg; }

  void addAllocaAdjustInst(MachineInstr* MI) {
    AllocaAdjustInsts.push_back(MI);
  }
  const std::vector<MachineInstr*>& getAllocaAdjustInsts() {
    return AllocaAdjustInsts;
  }

  void setVarArgsFrameIndex(int v) { VarArgsFrameIndex = v; }
  int getVarArgsFrameIndex() { return VarArgsFrameIndex; }

  void setStartPacket(MachineInstr* MI) {
    PacketInfo[MI] |= Hexagon::StartPacket;
  }
  void setEndPacket(MachineInstr* MI)   {
    PacketInfo[MI] |= Hexagon::EndPacket;
  }
  bool isStartPacket(const MachineInstr* MI) const {
    return (PacketInfo.count(MI) &&
            (PacketInfo.find(MI)->second & Hexagon::StartPacket));
  }
  bool isEndPacket(const MachineInstr* MI) const {
    return (PacketInfo.count(MI) &&
            (PacketInfo.find(MI)->second & Hexagon::EndPacket));
  }
  void setHasClobberLR(bool v) { HasClobberLR = v;  }
  bool hasClobberLR() const { return HasClobberLR; }

};
} // End llvm namespace

#endif
