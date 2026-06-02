//===-- EZHConstantPoolValue.h - EZH Constant Pool Value ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the EZHConstantPoolValue class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_EZH_EZHCONSTANTPOOLVALUE_H
#define LLVM_LIB_TARGET_EZH_EZHCONSTANTPOOLVALUE_H

#include "llvm/ADT/FoldingSet.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <vector>

namespace llvm {

class BlockAddress;
class GlobalValue;
class raw_ostream;

namespace EZHCP {
enum EZHCPKind {
  CPJumpTable,
  CPBlockAddress,
  CPExtSymbol,
  CPMachineBasicBlock,
  CPGlobalValue,
  CPConstantPoolIndex
};
}

/// EZH Constant Pool Value.
class EZHConstantPoolValue : public MachineConstantPoolValue {
  EZHCP::EZHCPKind Kind;
  unsigned JTI;                 // JumpTable index
  const BlockAddress *BA;       // BlockAddress value
  const char *Symbol;           // External symbol name
  const MachineBasicBlock *MBB; // MachineBasicBlock value
  const GlobalValue *GV;        // GlobalValue value
  int64_t Offset;               // Offset for GlobalValue
  unsigned CPI;                 // ConstantPoolIndex

public:
  // Constructors
  EZHConstantPoolValue(unsigned jti, Type *Ty)
      : MachineConstantPoolValue(Ty), Kind(EZHCP::CPJumpTable), JTI(jti),
        BA(nullptr), Symbol(nullptr), MBB(nullptr), GV(nullptr), Offset(0),
        CPI(0) {}

  EZHConstantPoolValue(const BlockAddress *ba, Type *Ty)
      : MachineConstantPoolValue(Ty), Kind(EZHCP::CPBlockAddress), JTI(0),
        BA(ba), Symbol(nullptr), MBB(nullptr), GV(nullptr), Offset(0), CPI(0) {}

  EZHConstantPoolValue(const char *sym, Type *Ty)
      : MachineConstantPoolValue(Ty), Kind(EZHCP::CPExtSymbol), JTI(0),
        BA(nullptr), Symbol(sym), MBB(nullptr), GV(nullptr), Offset(0), CPI(0) {
  }

  EZHConstantPoolValue(const MachineBasicBlock *mbb, Type *Ty)
      : MachineConstantPoolValue(Ty), Kind(EZHCP::CPMachineBasicBlock), JTI(0),
        BA(nullptr), Symbol(nullptr), MBB(mbb), GV(nullptr), Offset(0), CPI(0) {
  }

  EZHConstantPoolValue(const GlobalValue *gv, int64_t offset, Type *Ty)
      : MachineConstantPoolValue(Ty), Kind(EZHCP::CPGlobalValue), JTI(0),
        BA(nullptr), Symbol(nullptr), MBB(nullptr), GV(gv), Offset(offset),
        CPI(0) {}

  EZHConstantPoolValue(unsigned cpi, Type *Ty, bool isCPI)
      : MachineConstantPoolValue(Ty), Kind(EZHCP::CPConstantPoolIndex), JTI(0),
        BA(nullptr), Symbol(nullptr), MBB(nullptr), GV(nullptr), Offset(0),
        CPI(cpi) {}

  // Accessors
  bool isJumpTable() const { return Kind == EZHCP::CPJumpTable; }
  bool isBlockAddress() const { return Kind == EZHCP::CPBlockAddress; }
  bool isExtSymbol() const { return Kind == EZHCP::CPExtSymbol; }
  bool isMachineBasicBlock() const {
    return Kind == EZHCP::CPMachineBasicBlock;
  }
  bool isGlobalValue() const { return Kind == EZHCP::CPGlobalValue; }
  bool isConstantPoolIndex() const {
    return Kind == EZHCP::CPConstantPoolIndex;
  }

  unsigned getJumpTableIndex() const { return JTI; }
  const BlockAddress *getBlockAddress() const { return BA; }
  const char *getExtSymbol() const { return Symbol; }
  const MachineBasicBlock *getMachineBasicBlock() const { return MBB; }
  const GlobalValue *getGlobalValue() const { return GV; }
  int64_t getOffset() const { return Offset; }
  unsigned getConstantPoolIndex() const { return CPI; }

  // Virtual methods from MachineConstantPoolValue
  int getExistingMachineCPValue(MachineConstantPool *MCP,
                                Align Alignment) override {
    const std::vector<MachineConstantPoolEntry> &Constants =
        MCP->getConstants();
    for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
      if (Constants[i].isMachineConstantPoolEntry()) {
        auto *CPV =
            static_cast<EZHConstantPoolValue *>(Constants[i].Val.MachineCPVal);
        if (CPV->Kind == Kind && CPV->JTI == JTI && CPV->BA == BA &&
            CPV->Symbol == Symbol && CPV->MBB == MBB && CPV->GV == GV &&
            CPV->Offset == Offset && CPV->CPI == CPI)
          return i;
      }
    }
    return -1;
  }

  void print(raw_ostream &O) const override {
    switch (Kind) {
    case EZHCP::CPJumpTable:
      O << ".LJTI" << JTI;
      break;
    case EZHCP::CPBlockAddress:
      O << BA->getBasicBlock()->getName();
      break;
    case EZHCP::CPExtSymbol:
      O << Symbol;
      break;
    case EZHCP::CPMachineBasicBlock:
      O << MBB->getSymbol()->getName();
      break;
    case EZHCP::CPGlobalValue:
      O << "Global: " << static_cast<const void *>(GV) << "+" << Offset;
      break;
    case EZHCP::CPConstantPoolIndex:
      O << ".LCPI" << CPI;
      break;
    }
  }

  void addSelectionDAGCSEId(FoldingSetNodeID &ID) override {
    ID.AddInteger(Kind);
    ID.AddInteger(JTI);
    ID.AddPointer(BA);
    if (Symbol)
      ID.AddString(Symbol);
    ID.AddPointer(MBB);
    ID.AddPointer(GV);
    ID.AddInteger(Offset);
    ID.AddInteger(CPI);
  }

  static bool classof(const MachineConstantPoolValue *V) {
    return true; // Assume all are ours for now
  }
};

} // namespace llvm

#endif
