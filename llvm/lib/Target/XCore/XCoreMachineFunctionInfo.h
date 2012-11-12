//===-- XCoreMachineFuctionInfo.h - XCore machine function info -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares XCore-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#ifndef XCOREMACHINEFUNCTIONINFO_H
#define XCOREMACHINEFUNCTIONINFO_H

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include <vector>

namespace llvm {

// Forward declarations
class Function;

/// XCoreFunctionInfo - This class is derived from MachineFunction private
/// XCore target-specific information for each MachineFunction.
class XCoreFunctionInfo : public MachineFunctionInfo {
  virtual void anchor();
  bool UsesLR;
  int LRSpillSlot;
  int FPSpillSlot;
  int VarArgsFrameIndex;
  std::vector<std::pair<MCSymbol*, CalleeSavedInfo> > SpillLabels;

public:
  XCoreFunctionInfo() :
    UsesLR(false),
    LRSpillSlot(0),
    FPSpillSlot(0),
    VarArgsFrameIndex(0) {}
  
  explicit XCoreFunctionInfo(MachineFunction &MF) :
    UsesLR(false),
    LRSpillSlot(0),
    FPSpillSlot(0),
    VarArgsFrameIndex(0) {}
  
  ~XCoreFunctionInfo() {}
  
  void setVarArgsFrameIndex(int off) { VarArgsFrameIndex = off; }
  int getVarArgsFrameIndex() const { return VarArgsFrameIndex; }
  
  void setUsesLR(bool val) { UsesLR = val; }
  bool getUsesLR() const { return UsesLR; }
  
  void setLRSpillSlot(int off) { LRSpillSlot = off; }
  int getLRSpillSlot() const { return LRSpillSlot; }
  
  void setFPSpillSlot(int off) { FPSpillSlot = off; }
  int getFPSpillSlot() const { return FPSpillSlot; }
  
  std::vector<std::pair<MCSymbol*, CalleeSavedInfo> > &getSpillLabels() {
    return SpillLabels;
  }
};
} // End llvm namespace

#endif // XCOREMACHINEFUNCTIONINFO_H
