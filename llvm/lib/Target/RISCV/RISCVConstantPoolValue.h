//===--- RISCVConstantPoolValue.h - RISC-V constantpool value ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RISC-V specific constantpool value class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVCONSTANTPOOLVALUE_H
#define LLVM_LIB_TARGET_RISCV_RISCVCONSTANTPOOLVALUE_H

#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

class LLVMContext;
class GlobalValue;
class BlockAddress;

namespace RISCVCP {

enum RISCVCPKind { ExtSymbol, GlobalValue, BlockAddress };

enum RISCVCPModifier {
  None,
};
} // end namespace RISCVCP

/// A RISCV-specific constant pool value.
class RISCVConstantPoolValue : public MachineConstantPoolValue {
  RISCVCP::RISCVCPKind Kind;
  RISCVCP::RISCVCPModifier Modifier;

protected:
  RISCVConstantPoolValue(LLVMContext &C, RISCVCP::RISCVCPKind Kind,
                         RISCVCP::RISCVCPModifier Modifier);

  RISCVConstantPoolValue(Type *Ty, RISCVCP::RISCVCPKind Kind,
                         RISCVCP::RISCVCPModifier Modifier);

  template <typename Derived>
  int getExistingMachineCPValueImpl(MachineConstantPool *CP, Align Alignment) {
    const std::vector<MachineConstantPoolEntry> &Constants = CP->getConstants();
    for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
      if (Constants[i].isMachineConstantPoolEntry() &&
          Constants[i].getAlign() >= Alignment) {
        auto *CPV = static_cast<RISCVConstantPoolValue *>(
            Constants[i].Val.MachineCPVal);
        if (Derived *APC = dyn_cast<Derived>(CPV))
          if (cast<Derived>(this)->equals(APC))
            return i;
      }
    }

    return -1;
  }

public:
  ~RISCVConstantPoolValue() = default;

  RISCVCP::RISCVCPModifier getModifier() const { return Modifier; }
  StringRef getModifierText() const;
  bool hasModifier() const { return Modifier != RISCVCP::None; }

  bool isExtSymbol() const { return Kind == RISCVCP::ExtSymbol; }
  bool isGlobalValue() const { return Kind == RISCVCP::GlobalValue; }
  bool isBlockAddress() const { return Kind == RISCVCP::BlockAddress; }

  int getExistingMachineCPValue(MachineConstantPool *CP,
                                Align Alignment) override;

  void addSelectionDAGCSEId(FoldingSetNodeID &ID) override {}

  bool equals(const RISCVConstantPoolValue *A) const {
    return this->Modifier == A->Modifier;
  }

  void print(raw_ostream &O) const override;
};

class RISCVConstantPoolConstant : public RISCVConstantPoolValue {
  const Constant *CVal;

  RISCVConstantPoolConstant(Type *Ty, const Constant *GV,
                            RISCVCP::RISCVCPKind Kind);

public:
  static RISCVConstantPoolConstant *Create(const GlobalValue *GV,
                                           RISCVCP::RISCVCPKind Kind);
  static RISCVConstantPoolConstant *Create(const Constant *C,
                                           RISCVCP::RISCVCPKind Kind);

  const GlobalValue *getGlobalValue() const;
  const BlockAddress *getBlockAddress() const;

  int getExistingMachineCPValue(MachineConstantPool *CP,
                                Align Alignment) override;

  void addSelectionDAGCSEId(FoldingSetNodeID &ID) override;

  void print(raw_ostream &O) const override;

  bool equals(const RISCVConstantPoolConstant *A) const {
    return CVal == A->CVal && RISCVConstantPoolValue::equals(A);
  }

  static bool classof(const RISCVConstantPoolValue *RCPV) {
    return RCPV->isGlobalValue() || RCPV->isBlockAddress();
  }
};

class RISCVConstantPoolSymbol : public RISCVConstantPoolValue {
  const std::string S;

  RISCVConstantPoolSymbol(LLVMContext &C, StringRef s,
                          RISCVCP::RISCVCPModifier Modifier);

public:
  static RISCVConstantPoolSymbol *Create(LLVMContext &C, StringRef s,
                                         RISCVCP ::RISCVCPModifier Modifier);

  std::string getSymbol() const { return S; }

  int getExistingMachineCPValue(MachineConstantPool *CP,
                                Align Alignment) override;

  void addSelectionDAGCSEId(FoldingSetNodeID &ID) override;

  void print(raw_ostream &O) const override;

  bool equals(const RISCVConstantPoolSymbol *A) const {
    return S == A->S && RISCVConstantPoolValue::equals(A);
  }
  static bool classof(const RISCVConstantPoolValue *RCPV) {
    return RCPV->isExtSymbol();
  }
};

} // end namespace llvm

#endif
