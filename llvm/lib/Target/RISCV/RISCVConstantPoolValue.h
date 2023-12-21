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

#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

class BlockAddress;
class GlobalValue;
class LLVMContext;

/// A RISCV-specific constant pool value.
class RISCVConstantPoolValue : public MachineConstantPoolValue {
protected:
  enum class RISCVCPKind { ExtSymbol, GlobalValue, BlockAddress };

  RISCVConstantPoolValue(LLVMContext &C, RISCVCPKind Kind);

  RISCVConstantPoolValue(Type *Ty, RISCVCPKind Kind);

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

private:
  RISCVCPKind Kind;

public:
  ~RISCVConstantPoolValue() = default;

  bool isExtSymbol() const { return Kind == RISCVCPKind::ExtSymbol; }
  bool isGlobalValue() const { return Kind == RISCVCPKind::GlobalValue; }
  bool isBlockAddress() const { return Kind == RISCVCPKind::BlockAddress; }

  int getExistingMachineCPValue(MachineConstantPool *CP,
                                Align Alignment) override;

  void addSelectionDAGCSEId(FoldingSetNodeID &ID) override {}
};

class RISCVConstantPoolConstant : public RISCVConstantPoolValue {
  const Constant *CVal;

  RISCVConstantPoolConstant(Type *Ty, const Constant *GV, RISCVCPKind Kind);

public:
  static RISCVConstantPoolConstant *Create(const GlobalValue *GV);
  static RISCVConstantPoolConstant *Create(const BlockAddress *BA);

  const GlobalValue *getGlobalValue() const;
  const BlockAddress *getBlockAddress() const;

  int getExistingMachineCPValue(MachineConstantPool *CP,
                                Align Alignment) override;

  void addSelectionDAGCSEId(FoldingSetNodeID &ID) override;

  void print(raw_ostream &O) const override;

  bool equals(const RISCVConstantPoolConstant *A) const {
    return CVal == A->CVal;
  }

  static bool classof(const RISCVConstantPoolValue *RCPV) {
    return RCPV->isGlobalValue() || RCPV->isBlockAddress();
  }
};

class RISCVConstantPoolSymbol : public RISCVConstantPoolValue {
  const StringRef S;

  RISCVConstantPoolSymbol(LLVMContext &C, StringRef s);

public:
  static RISCVConstantPoolSymbol *Create(LLVMContext &C, StringRef s);

  StringRef getSymbol() const { return S; }

  int getExistingMachineCPValue(MachineConstantPool *CP,
                                Align Alignment) override;

  void addSelectionDAGCSEId(FoldingSetNodeID &ID) override;

  void print(raw_ostream &O) const override;

  bool equals(const RISCVConstantPoolSymbol *A) const { return S == A->S; }

  static bool classof(const RISCVConstantPoolValue *RCPV) {
    return RCPV->isExtSymbol();
  }
};

} // end namespace llvm

#endif
