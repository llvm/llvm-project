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
  const GlobalValue *GV;
  const StringRef S;

  RISCVConstantPoolValue(Type *Ty, const GlobalValue *GV);
  RISCVConstantPoolValue(LLVMContext &C, StringRef S);

private:
  enum class RISCVCPKind { ExtSymbol, GlobalValue };
  RISCVCPKind Kind;

public:
  ~RISCVConstantPoolValue() = default;

  static RISCVConstantPoolValue *Create(const GlobalValue *GV);
  static RISCVConstantPoolValue *Create(LLVMContext &C, StringRef S);

  bool isGlobalValue() const { return Kind == RISCVCPKind::GlobalValue; }
  bool isExtSymbol() const { return Kind == RISCVCPKind::ExtSymbol; }

  const GlobalValue *getGlobalValue() const { return GV; }
  StringRef getSymbol() const { return S; }

  int getExistingMachineCPValue(MachineConstantPool *CP,
                                Align Alignment) override;

  void addSelectionDAGCSEId(FoldingSetNodeID &ID) override;

  void print(raw_ostream &O) const override;

  bool equals(const RISCVConstantPoolValue *A) const;
};

} // end namespace llvm

#endif
