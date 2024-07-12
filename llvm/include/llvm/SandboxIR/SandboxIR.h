//===- SandboxIR.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Sandbox IR is a lightweight overlay transactional IR on top of LLVM IR.
// Features:
// - You can save/rollback the state of the IR at any time.
// - Any changes made to Sandbox IR will automatically update the underlying
//   LLVM IR so both IRs are always in sync.
// - Feels like LLVM IR, similar API.
//
// SandboxIR forms a class hierarchy that resembles that of LLVM IR
// but is in the `sandboxir` namespace:
//
// namespace sandboxir {
//
//        +- Argument                   +- BinaryOperator
//        |                             |
// Value -+- BasicBlock                 +- BranchInst
//        |                             |
//        +- Function   +- Constant     +- CastInst
//        |             |               |
//        +- User ------+- Instruction -+- CallInst
//                                      |
//                                      +- CmpInst
//                                      |
//                                      +- ExtractElementInst
//                                      |
//                                      +- GetElementPtrInst
//                                      |
//                                      +- InsertElementInst
//                                      |
//                                      +- LoadInst
//                                      |
//                                      +- OpaqueInst
//                                      |
//                                      +- PHINode
//                                      |
//                                      +- RetInst
//                                      |
//                                      +- SelectInst
//                                      |
//                                      +- ShuffleVectorInst
//                                      |
//                                      +- StoreInst
//                                      |
//                                      +- UnaryOperator
//
// Use
//
// } // namespace sandboxir
//

#ifndef LLVM_TRANSFORMS_SANDBOXIR_SANDBOXIR_H
#define LLVM_TRANSFORMS_SANDBOXIR_SANDBOXIR_H

#include "llvm/IR/Function.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

namespace sandboxir {

class Context;

/// A SandboxIR Value has users. This is the base class.
class Value {
public:
  enum class ClassID : unsigned {
#define DEF_VALUE(ID, CLASS) ID,
#define DEF_USER(ID, CLASS) ID,
#define DEF_INSTR(ID, OPC, CLASS) ID,
#include "llvm/SandboxIR/SandboxIRValues.def"
  };

protected:
  static const char *getSubclassIDStr(ClassID ID) {
    switch (ID) {
#define DEF_VALUE(ID, CLASS)                                                   \
  case ClassID::ID:                                                            \
    return #ID;
#define DEF_USER(ID, CLASS)                                                    \
  case ClassID::ID:                                                            \
    return #ID;
#define DEF_INSTR(ID, OPC, CLASS)                                              \
  case ClassID::ID:                                                            \
    return #ID;
#include "llvm/SandboxIR/SandboxIRValues.def"
    }
    llvm_unreachable("Unimplemented ID");
  }

  /// For isa/dyn_cast.
  ClassID SubclassID;
#ifndef NDEBUG
  /// A unique ID used for forming the name (used for debugging).
  unsigned UID;
#endif
  /// The LLVM Value that corresponds to this SandboxIR Value.
  /// NOTE: Some SBInstructions, like Packs, may include more than one value.
  llvm::Value *Val = nullptr;

  /// All values point to the context.
  Context &Ctx;
  // This is used by eraseFromParent().
  void clearValue() { Val = nullptr; }
  template <typename ItTy, typename SBTy> friend class LLVMOpUserItToSBTy;

public:
  Value(ClassID SubclassID, llvm::Value *Val, Context &Ctx);
  virtual ~Value() = default;
  ClassID getSubclassID() const { return SubclassID; }

  Type *getType() const { return Val->getType(); }

  Context &getContext() const;
#ifndef NDEBUG
  /// Should crash if there is something wrong with the instruction.
  virtual void verify() const = 0;
  /// Returns the name in the form 'SB<number>.' like 'SB1.'
  std::string getName() const;
  virtual void dumpCommonHeader(raw_ostream &OS) const;
  void dumpCommonFooter(raw_ostream &OS) const;
  void dumpCommonPrefix(raw_ostream &OS) const;
  void dumpCommonSuffix(raw_ostream &OS) const;
  void printAsOperandCommon(raw_ostream &OS) const;
  friend raw_ostream &operator<<(raw_ostream &OS, const sandboxir::Value &V) {
    V.dump(OS);
    return OS;
  }
  virtual void dump(raw_ostream &OS) const = 0;
  LLVM_DUMP_METHOD virtual void dump() const = 0;
#endif
};

/// Argument of a sandboxir::Function.
class Argument : public sandboxir::Value {
public:
  Argument(llvm::Argument *Arg, sandboxir::Context &Ctx)
      : sandboxir::Value(ClassID::Argument, Arg, Ctx) {}
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::Argument;
  }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::Argument>(Val) && "Expected Argument!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const sandboxir::Argument &TArg) {
    TArg.dump(OS);
    return OS;
  }
  void printAsOperand(raw_ostream &OS) const;
  void dump(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dump() const final;
#endif
};

class User : public Value {
public:
  User(ClassID ID, llvm::Value *V, Context &Ctx) : Value(ID, V, Ctx) {}
  /// For isa/dyn_cast.
  static bool classof(const Value *From);
#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::User>(Val) && "Expected User!");
  }
  void dumpCommonHeader(raw_ostream &OS) const final;
  void dump(raw_ostream &OS) const override {
    // TODO: Remove this tmp implementation once we get the Instruction classes.
  }
  LLVM_DUMP_METHOD void dump() const override {
    // TODO: Remove this tmp implementation once we get the Instruction classes.
  }
#endif
};

class Constant : public sandboxir::User {
public:
  Constant(llvm::Constant *C, sandboxir::Context &SBCtx)
      : sandboxir::User(ClassID::Constant, C, SBCtx) {}
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::Constant ||
           From->getSubclassID() == ClassID::Function;
  }
  sandboxir::Context &getParent() const { return getContext(); }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::Constant>(Val) && "Expected Constant!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const sandboxir::Constant &SBC) {
    SBC.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
#endif
};

/// A sandboxir::User with operands and opcode.
class Instruction : public sandboxir::User {
public:
  enum class Opcode {
#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)
#define OP(OPC) OPC,
#define DEF_INSTR(ID, OPC, CLASS) OPC
#include "llvm/SandboxIR/SandboxIRValues.def"
  };

  Instruction(ClassID ID, Opcode Opc, llvm::Instruction *I,
              sandboxir::Context &SBCtx)
      : sandboxir::User(ID, I, SBCtx), Opc(Opc) {}

protected:
  Opcode Opc;

public:
  static const char *getOpcodeName(Opcode Opc);
#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS, Opcode Opc) {
    OS << getOpcodeName(Opc);
    return OS;
  }
#endif
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From);

#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const sandboxir::Instruction &SBI) {
    SBI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
#endif
};

/// An LLLVM Instruction that has no SandboxIR equivalent class gets mapped to
/// an OpaqueInstr.
class OpaqueInst : public sandboxir::Instruction {
public:
  OpaqueInst(llvm::Instruction *I, sandboxir::Context &Ctx)
      : sandboxir::Instruction(ClassID::Opaque, Opcode::Opaque, I, Ctx) {}
  OpaqueInst(ClassID SubclassID, llvm::Instruction *I, sandboxir::Context &Ctx)
      : sandboxir::Instruction(SubclassID, Opcode::Opaque, I, Ctx) {}
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::Opaque;
  }
#ifndef NDEBUG
  void verify() const final {
    // Nothing to do
  }
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const sandboxir::OpaqueInst &OI) {
    OI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
#endif
};

class Context {
protected:
  LLVMContext &LLVMCtx;
  /// Maps LLVM Value to the corresponding sandboxir::Value. Owns all
  /// SandboxIR objects.
  DenseMap<llvm::Value *, std::unique_ptr<sandboxir::Value>>
      LLVMValueToValueMap;

public:
  Context(LLVMContext &LLVMCtx) : LLVMCtx(LLVMCtx) {}
  sandboxir::Value *getValue(llvm::Value *V) const;
};

class Function : public sandboxir::Value {
public:
  Function(llvm::Function *F, sandboxir::Context &Ctx)
      : sandboxir::Value(ClassID::Function, F, Ctx) {}
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::Function;
  }

#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::Function>(Val) && "Expected Function!");
  }
  void dumpNameAndArgs(raw_ostream &OS) const;
  void dump(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dump() const final;
#endif
};

} // namespace sandboxir
} // namespace llvm

#endif // LLVM_TRANSFORMS_SANDBOXIR_SANDBOXIR_H
