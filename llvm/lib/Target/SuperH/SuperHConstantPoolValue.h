//===- SuperHConstantPoolValue.h - SuperH constantpool value ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SuperH specific constantpool value class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SUPERH_SUPERHCONSTANTPOOLVALUE_H
#define LLVM_LIB_TARGET_SUPERH_SUPERHCONSTANTPOOLVALUE_H

#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/MC/MCSymbol.h"

namespace llvm {

class BlockAddress;
class Constant;
class GlobalValue;
class GlobalVariable;
class LLVMContext;
class MachineBasicBlock;
class raw_ostream;
class Type;

namespace SHCP {

  enum SHCPKind {
    CPValue,
    CPExtSymbol,
    CPBlockAddress,
    CPMachineBasicBlock,
    CPPromotedGlobal
  };

  enum SHCPModifier {
    no_modifier,  /// None
    DIR,          /// Direct
    GOT_PCREL,    /// Global Offset Table, PC Relative
    GOT_PLTOFF,   /// Global Offset Table, Thread Pointer Offset
  };

} // end namespace SHCP

class SuperHConstantPoolValue : public MachineConstantPoolValue {
  unsigned LabelId;           // Label id of the load.
  SHCP::SHCPKind Kind;      // Kind of constant.
  SHCP::SHCPModifier Modifier;  // GV modifier i.e. (&GV(modifier)-(LPIC+8))
protected:
  SuperHConstantPoolValue(Type *Ty, unsigned id, SHCP::SHCPKind Kind, 
                          SHCP::SHCPModifier Modifier);

  SuperHConstantPoolValue(LLVMContext &C, unsigned id, SHCP::SHCPKind Kind,
                          SHCP::SHCPModifier Modifier);

  template <typename Derived>
  int getExistingMachineCPValueImpl(MachineConstantPool *CP, Align Alignment) {
    const std::vector<MachineConstantPoolEntry> &Constants = CP->getConstants();
    for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
      if (Constants[i].isMachineConstantPoolEntry() &&
          Constants[i].getAlign() >= Alignment) {
        auto *CPV =
          static_cast<SuperHConstantPoolValue*>(Constants[i].Val.MachineCPVal);
        if (Derived *APC = dyn_cast<Derived>(CPV))
          if (cast<Derived>(this)->equals(APC))
            return i;
      }
    }

    return -1;
  }

public:
  ~SuperHConstantPoolValue() override;

  SHCP::SHCPKind getKind() const { return Kind; }
  SHCP::SHCPModifier getModifier() const { return Modifier; }
  StringRef getModifierText() const;
  bool hasModifier() const { return Modifier != SHCP::no_modifier; }

  unsigned getLabelId() const { return LabelId; }

  bool isGlobalValue() const { return Kind == SHCP::CPValue; }
  bool isExtSymbol() const { return Kind == SHCP::CPExtSymbol; }
  bool isBlockAddress() const { return Kind == SHCP::CPBlockAddress; }
  bool isMachineBasicBlock() const{ return Kind == SHCP::CPMachineBasicBlock; }
  bool isPromotedGlobal() const{ return Kind == SHCP::CPPromotedGlobal; }

  int getExistingMachineCPValue(MachineConstantPool *CP,
                                Align Alignment) override;

  void addSelectionDAGCSEId(FoldingSetNodeID &ID) override;

  /// hasSameValue - Return true if this ARM constpool value can share the same
  /// constantpool entry as another ARM constpool value.
  virtual bool hasSameValue(SuperHConstantPoolValue *ACPV);

  bool equals(const SuperHConstantPoolValue *A) const {
    return this->LabelId == A->LabelId &&
      this->Modifier == A->Modifier;
  }

  void print(raw_ostream &O) const override;
  void print(raw_ostream *O) const { if (O) print(*O); }
  void dump() const;
};

inline raw_ostream &operator<<(raw_ostream &O, const SuperHConstantPoolValue &V) {
  V.print(O);
  return O;
}


/// SuperHConstantPoolConstant - SuperH-specific constant pool values for Constants,
/// Functions, and BlockAddresses.
class SuperHConstantPoolConstant : public SuperHConstantPoolValue {
  const Constant *CVal;         // Constant being loaded.
  SmallPtrSet<const GlobalVariable*, 1> GVars;

  SuperHConstantPoolConstant(const Constant *C,
                             unsigned ID,
                             SHCP::SHCPKind Kind,
                             SHCP::SHCPModifier Modifier);
  SuperHConstantPoolConstant(Type *Ty, const Constant *C,
                             unsigned ID,
                             SHCP::SHCPKind Kind,
                             SHCP::SHCPModifier Modifier);
  SuperHConstantPoolConstant(const GlobalVariable *GV, const Constant *Init);

public:
  static SuperHConstantPoolConstant *Create(const Constant *C, unsigned ID);
  static SuperHConstantPoolConstant *Create(const GlobalValue *GV,
                                         SHCP::SHCPModifier Modifier);
  static SuperHConstantPoolConstant *Create(const GlobalVariable *GV,
                                         const Constant *Initializer);
  static SuperHConstantPoolConstant *Create(const Constant *C, unsigned ID,
                                         SHCP::SHCPKind Kind);
  static SuperHConstantPoolConstant *Create(const Constant *C, unsigned ID,
                                         SHCP::SHCPKind Kind,
                                         SHCP::SHCPModifier Modifier);

  const GlobalValue *getGV() const;
  const BlockAddress *getBlockAddress() const;

  using promoted_iterator = SmallPtrSet<const GlobalVariable *, 1>::iterator;

  iterator_range<promoted_iterator> promotedGlobals() { return GVars; }

  const Constant *getPromotedGlobalInit() const {
    return CVal;
  }

  int getExistingMachineCPValue(MachineConstantPool *CP,
                                Align Alignment) override;

  /// hasSameValue - Return true if this ARM constpool value can share the same
  /// constantpool entry as another ARM constpool value.
  bool hasSameValue(SuperHConstantPoolValue *ACPV) override;

  void addSelectionDAGCSEId(FoldingSetNodeID &ID) override;

  void print(raw_ostream &O) const override;

  static bool classof(const SuperHConstantPoolValue *APV) {
    return APV->isGlobalValue() || APV->isBlockAddress() ||
           APV->isPromotedGlobal();
  }

  bool equals(const SuperHConstantPoolConstant *A) const {
    return CVal == A->CVal && SuperHConstantPoolValue::equals(A);
  }
};

/// SuperHConstantPoolSymbol - SH-specific constantpool 
/// values for external symbols.
class SuperHConstantPoolSymbol : public SuperHConstantPoolValue {
  const std::string S;          // ExtSymbol being loaded.

  SuperHConstantPoolSymbol(LLVMContext &C, StringRef s, unsigned id, SHCP::SHCPModifier Modifier);

public:
  static SuperHConstantPoolSymbol *Create(LLVMContext &C, StringRef s, unsigned ID);

  StringRef getSymbol() const { return S; }

  int getExistingMachineCPValue(MachineConstantPool *CP,
                                Align Alignment) override;

  void addSelectionDAGCSEId(FoldingSetNodeID &ID) override;

  /// hasSameValue - Return true if this ARM constpool value can share the same
  /// constantpool entry as another ARM constpool value.
  bool hasSameValue(SuperHConstantPoolValue *SCPV) override;

  void print(raw_ostream &O) const override;

  static bool classof(const SuperHConstantPoolValue *SCPV) {
    return SCPV->isExtSymbol();
  }

  bool equals(const SuperHConstantPoolSymbol *A) const {
    return S == A->S && SuperHConstantPoolValue::equals(A);
  }
};

} // namespace llvm

#endif