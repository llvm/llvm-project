//===- llvm/InlineAsm.h - Class to represent inline asm strings -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class represents the inline asm strings, which are Value*'s that are
// used as the callee operand of call instructions.  InlineAsm's are uniqued
// like constants, and created via InlineAsm::get(...).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_INLINEASM_H
#define LLVM_IR_INLINEASM_H

#include "llvm/ADT/Bitfields.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <string>
#include <vector>

namespace llvm {

class Error;
class FunctionType;
class PointerType;
template <class ConstantClass> class ConstantUniqueMap;

class InlineAsm final : public Value {
public:
  enum AsmDialect {
    AD_ATT,
    AD_Intel
  };

private:
  friend struct InlineAsmKeyType;
  friend class ConstantUniqueMap<InlineAsm>;

  std::string AsmString, Constraints;
  FunctionType *FTy;
  bool HasSideEffects;
  bool IsAlignStack;
  AsmDialect Dialect;
  bool CanThrow;

  InlineAsm(FunctionType *Ty, const std::string &AsmString,
            const std::string &Constraints, bool hasSideEffects,
            bool isAlignStack, AsmDialect asmDialect, bool canThrow);

  /// When the ConstantUniqueMap merges two types and makes two InlineAsms
  /// identical, it destroys one of them with this method.
  void destroyConstant();

public:
  InlineAsm(const InlineAsm &) = delete;
  InlineAsm &operator=(const InlineAsm &) = delete;

  /// InlineAsm::get - Return the specified uniqued inline asm string.
  ///
  static InlineAsm *get(FunctionType *Ty, StringRef AsmString,
                        StringRef Constraints, bool hasSideEffects,
                        bool isAlignStack = false,
                        AsmDialect asmDialect = AD_ATT, bool canThrow = false);

  bool hasSideEffects() const { return HasSideEffects; }
  bool isAlignStack() const { return IsAlignStack; }
  AsmDialect getDialect() const { return Dialect; }
  bool canThrow() const { return CanThrow; }

  /// getType - InlineAsm's are always pointers.
  ///
  PointerType *getType() const {
    return reinterpret_cast<PointerType*>(Value::getType());
  }

  /// getFunctionType - InlineAsm's are always pointers to functions.
  ///
  FunctionType *getFunctionType() const;

  const std::string &getAsmString() const { return AsmString; }
  const std::string &getConstraintString() const { return Constraints; }
  void collectAsmStrs(SmallVectorImpl<StringRef> &AsmStrs) const;

  /// This static method can be used by the parser to check to see if the
  /// specified constraint string is legal for the type.
  static Error verify(FunctionType *Ty, StringRef Constraints);

  // Constraint String Parsing
  enum ConstraintPrefix {
    isInput,            // 'x'
    isOutput,           // '=x'
    isClobber,          // '~x'
    isLabel,            // '!x'
  };

  using ConstraintCodeVector = std::vector<std::string>;

  struct SubConstraintInfo {
    /// MatchingInput - If this is not -1, this is an output constraint where an
    /// input constraint is required to match it (e.g. "0").  The value is the
    /// constraint number that matches this one (for example, if this is
    /// constraint #0 and constraint #4 has the value "0", this will be 4).
    int MatchingInput = -1;

    /// Code - The constraint code, either the register name (in braces) or the
    /// constraint letter/number.
    ConstraintCodeVector Codes;

    /// Default constructor.
    SubConstraintInfo() = default;
  };

  using SubConstraintInfoVector = std::vector<SubConstraintInfo>;
  struct ConstraintInfo;
  using ConstraintInfoVector = std::vector<ConstraintInfo>;

  struct ConstraintInfo {
    /// Type - The basic type of the constraint: input/output/clobber/label
    ///
    ConstraintPrefix Type = isInput;

    /// isEarlyClobber - "&": output operand writes result before inputs are all
    /// read.  This is only ever set for an output operand.
    bool isEarlyClobber = false;

    /// MatchingInput - If this is not -1, this is an output constraint where an
    /// input constraint is required to match it (e.g. "0").  The value is the
    /// constraint number that matches this one (for example, if this is
    /// constraint #0 and constraint #4 has the value "0", this will be 4).
    int MatchingInput = -1;

    /// hasMatchingInput - Return true if this is an output constraint that has
    /// a matching input constraint.
    bool hasMatchingInput() const { return MatchingInput != -1; }

    /// isCommutative - This is set to true for a constraint that is commutative
    /// with the next operand.
    bool isCommutative = false;

    /// isIndirect - True if this operand is an indirect operand.  This means
    /// that the address of the source or destination is present in the call
    /// instruction, instead of it being returned or passed in explicitly.  This
    /// is represented with a '*' in the asm string.
    bool isIndirect = false;

    /// Code - The constraint code, either the register name (in braces) or the
    /// constraint letter/number.
    ConstraintCodeVector Codes;

    /// isMultipleAlternative - '|': has multiple-alternative constraints.
    bool isMultipleAlternative = false;

    /// multipleAlternatives - If there are multiple alternative constraints,
    /// this array will contain them.  Otherwise it will be empty.
    SubConstraintInfoVector multipleAlternatives;

    /// The currently selected alternative constraint index.
    unsigned currentAlternativeIndex = 0;

    /// Default constructor.
    ConstraintInfo() = default;

    /// Parse - Analyze the specified string (e.g. "=*&{eax}") and fill in the
    /// fields in this structure.  If the constraint string is not understood,
    /// return true, otherwise return false.
    bool Parse(StringRef Str, ConstraintInfoVector &ConstraintsSoFar);

    /// selectAlternative - Point this constraint to the alternative constraint
    /// indicated by the index.
    void selectAlternative(unsigned index);

    /// Whether this constraint corresponds to an argument.
    bool hasArg() const {
      return Type == isInput || (Type == isOutput && isIndirect);
    }
  };

  /// ParseConstraints - Split up the constraint string into the specific
  /// constraints and their prefixes.  If this returns an empty vector, and if
  /// the constraint string itself isn't empty, there was an error parsing.
  static ConstraintInfoVector ParseConstraints(StringRef ConstraintString);

  /// ParseConstraints - Parse the constraints of this inlineasm object,
  /// returning them the same way that ParseConstraints(str) does.
  ConstraintInfoVector ParseConstraints() const {
    return ParseConstraints(Constraints);
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const Value *V) {
    return V->getValueID() == Value::InlineAsmVal;
  }

  enum : uint32_t {
    // Fixed operands on an INLINEASM SDNode.
    Op_InputChain = 0,
    Op_AsmString = 1,
    Op_MDNode = 2,
    Op_ExtraInfo = 3, // HasSideEffects, IsAlignStack, AsmDialect.
    Op_FirstOperand = 4,

    // Fixed operands on an INLINEASM MachineInstr.
    MIOp_AsmString = 0,
    MIOp_ExtraInfo = 1, // HasSideEffects, IsAlignStack, AsmDialect.
    MIOp_FirstOperand = 2,

    // Interpretation of the MIOp_ExtraInfo bit field.
    Extra_HasSideEffects = 1,
    Extra_IsAlignStack = 2,
    Extra_AsmDialect = 4,
    Extra_MayLoad = 8,
    Extra_MayStore = 16,
    Extra_IsConvergent = 32,
  };

  // Inline asm operands map to multiple SDNode / MachineInstr operands.
  // The first operand is an immediate describing the asm operand, the low
  // bits is the kind:
  enum class Kind : uint8_t {
    RegUse = 1,             // Input register, "r".
    RegDef = 2,             // Output register, "=r".
    RegDefEarlyClobber = 3, // Early-clobber output register, "=&r".
    Clobber = 4,            // Clobbered register, "~r".
    Imm = 5,                // Immediate.
    Mem = 6,                // Memory operand, "m", or an address, "p".
    Func = 7,               // Address operand of function call
  };

  // Memory constraint codes.
  // Addresses are included here as they need to be treated the same by the
  // backend, the only difference is that they are not used to actaully
  // access memory by the instruction.
  enum class ConstraintCode : uint32_t {
    Unknown = 0,
    es,
    i,
    k,
    m,
    o,
    v,
    A,
    Q,
    R,
    S,
    T,
    Um,
    Un,
    Uq,
    Us,
    Ut,
    Uv,
    Uy,
    X,
    Z,
    ZB,
    ZC,
    Zy,

    // Address constraints
    p,
    ZQ,
    ZR,
    ZS,
    ZT,

    Max = ZT,
  };

  // These are helper methods for dealing with flags in the INLINEASM SDNode
  // in the backend.
  //
  // The encoding of Flag is currently:
  //   Bits 2-0 - A Kind::* value indicating the kind of the operand.
  //   Bits 15-3 - The number of SDNode operands associated with this inline
  //               assembly operand.
  //   If bit 31 is set:
  //     Bit 30-16 - The operand number that this operand must match.
  //                 When bits 2-0 are Kind::Mem, the Constraint_* value must be
  //                 obtained from the flags for this operand number.
  //   Else if bits 2-0 are Kind::Mem:
  //     Bit 30-16 - A Constraint_* value indicating the original constraint
  //                 code.
  //   Else:
  //     Bit 30-16 - The register class ID to use for the operand.
  //
  //  Bits 30-16 are called "Data" for lack of a better name. The getter is
  //  intentionally private; the public methods that rely on that private method
  //  should be used to check invariants first before accessing Data.
  class Flag {
    uint32_t Storage;
    using KindField = Bitfield::Element<Kind, 0, 3, Kind::Func>;
    using NumOperands = Bitfield::Element<unsigned, 3, 13>;
    using Data = Bitfield::Element<unsigned, 16, 15>;
    using IsMatched = Bitfield::Element<bool, 31, 1>;

    unsigned getData() const { return Bitfield::get<Data>(Storage); }
    bool isMatched() const { return Bitfield::get<IsMatched>(Storage); }
    void setKind(Kind K) { Bitfield::set<KindField>(Storage, K); }
    void setNumOperands(unsigned N) { Bitfield::set<NumOperands>(Storage, N); }
    void setData(unsigned D) { Bitfield::set<Data>(Storage, D); }
    void setIsMatched(bool B) { Bitfield::set<IsMatched>(Storage, B); }

  public:
    Flag() : Storage(0) {}
    explicit Flag(uint32_t F) : Storage(F) {}
    Flag(enum Kind K, unsigned NumOps) {
      setKind(K);
      setNumOperands(NumOps);
      setData(0);
      setIsMatched(false);
    }
    operator uint32_t() { return Storage; }
    Kind getKind() const { return Bitfield::get<KindField>(Storage); }
    bool isRegUseKind() const { return getKind() == Kind::RegUse; }
    bool isRegDefKind() const { return getKind() == Kind::RegDef; }
    bool isRegDefEarlyClobberKind() const {
      return getKind() == Kind::RegDefEarlyClobber;
    }
    bool isClobberKind() const { return getKind() == Kind::Clobber; }
    bool isImmKind() const { return getKind() == Kind::Imm; }
    bool isMemKind() const { return getKind() == Kind::Mem; }
    bool isFuncKind() const { return getKind() == Kind::Func; }
    StringRef getKindName() const {
      switch (getKind()) {
      case Kind::RegUse:
        return "reguse";
      case Kind::RegDef:
        return "regdef";
      case Kind::RegDefEarlyClobber:
        return "regdef-ec";
      case Kind::Clobber:
        return "clobber";
      case Kind::Imm:
        return "imm";
      case Kind::Mem:
      case Kind::Func:
        return "mem";
      }
      llvm_unreachable("impossible kind");
    }

    /// getNumOperandRegisters - Extract the number of registers field from the
    /// inline asm operand flag.
    unsigned getNumOperandRegisters() const {
      return Bitfield::get<NumOperands>(Storage);
    }

    /// isUseOperandTiedToDef - Return true if the flag of the inline asm
    /// operand indicates it is an use operand that's matched to a def operand.
    bool isUseOperandTiedToDef(unsigned &Idx) const {
      if (!isMatched())
        return false;
      Idx = getData();
      return true;
    }

    /// hasRegClassConstraint - Returns true if the flag contains a register
    /// class constraint.  Sets RC to the register class ID.
    bool hasRegClassConstraint(unsigned &RC) const {
      if (isMatched())
        return false;
      // setRegClass() uses 0 to mean no register class, and otherwise stores
      // RC + 1.
      if (!getData())
        return false;
      RC = getData() - 1;
      return true;
    }

    ConstraintCode getMemoryConstraintID() const {
      assert((isMemKind() || isFuncKind()) &&
             "Not expected mem or function flag!");
      uint32_t D = getData();
      assert(D < static_cast<uint32_t>(ConstraintCode::Max) &&
             D >= static_cast<uint32_t>(ConstraintCode::Unknown) &&
             "unexpected value for memory constraint");
      return static_cast<ConstraintCode>(D);
    }

    /// setMatchingOp - Augment an existing flag with information indicating
    /// that this input operand is tied to a previous output operand.
    void setMatchingOp(unsigned MatchedOperandNo) {
      assert(getData() == 0 && "Matching operand already set");
      setData(MatchedOperandNo);
      setIsMatched(true);
    }

    /// setRegClass - Augment an existing flag with the required register class
    /// for the following register operands. A tied use operand cannot have a
    /// register class, use the register class from the def operand instead.
    void setRegClass(unsigned RC) {
      assert(!isImmKind() && "Immediates cannot have a register class");
      assert(!isMemKind() && "Memory operand cannot have a register class");
      assert(getData() == 0 && "Register class already set");
      // Store RC + 1, reserve the value 0 to mean 'no register class'.
      setData(RC + 1);
    }

    /// setMemConstraint - Augment an existing flag with the constraint code for
    /// a memory constraint.
    void setMemConstraint(ConstraintCode C) {
      assert((isMemKind() || isFuncKind()) &&
             "Flag is not a memory or function constraint!");
      assert(getData() == 0 && "Mem constraint already set");
      setData(static_cast<uint32_t>(C));
    }
    /// clearMemConstraint - Similar to setMemConstraint(0), but without the
    /// assertion checking that the constraint has not been set previously.
    void clearMemConstraint() {
      assert((isMemKind() || isFuncKind()) &&
             "Flag is not a memory or function constraint!");
      setData(0);
    }
  };

  static std::vector<StringRef> getExtraInfoNames(unsigned ExtraInfo) {
    std::vector<StringRef> Result;
    if (ExtraInfo & InlineAsm::Extra_HasSideEffects)
      Result.push_back("sideeffect");
    if (ExtraInfo & InlineAsm::Extra_MayLoad)
      Result.push_back("mayload");
    if (ExtraInfo & InlineAsm::Extra_MayStore)
      Result.push_back("maystore");
    if (ExtraInfo & InlineAsm::Extra_IsConvergent)
      Result.push_back("isconvergent");
    if (ExtraInfo & InlineAsm::Extra_IsAlignStack)
      Result.push_back("alignstack");

    AsmDialect Dialect =
        InlineAsm::AsmDialect((ExtraInfo & InlineAsm::Extra_AsmDialect));

    if (Dialect == InlineAsm::AD_ATT)
      Result.push_back("attdialect");
    if (Dialect == InlineAsm::AD_Intel)
      Result.push_back("inteldialect");

    return Result;
  }

  static StringRef getMemConstraintName(ConstraintCode C) {
    switch (C) {
    case ConstraintCode::es:
      return "es";
    case ConstraintCode::i:
      return "i";
    case ConstraintCode::k:
      return "k";
    case ConstraintCode::m:
      return "m";
    case ConstraintCode::o:
      return "o";
    case ConstraintCode::v:
      return "v";
    case ConstraintCode::A:
      return "A";
    case ConstraintCode::Q:
      return "Q";
    case ConstraintCode::R:
      return "R";
    case ConstraintCode::S:
      return "S";
    case ConstraintCode::T:
      return "T";
    case ConstraintCode::Um:
      return "Um";
    case ConstraintCode::Un:
      return "Un";
    case ConstraintCode::Uq:
      return "Uq";
    case ConstraintCode::Us:
      return "Us";
    case ConstraintCode::Ut:
      return "Ut";
    case ConstraintCode::Uv:
      return "Uv";
    case ConstraintCode::Uy:
      return "Uy";
    case ConstraintCode::X:
      return "X";
    case ConstraintCode::Z:
      return "Z";
    case ConstraintCode::ZB:
      return "ZB";
    case ConstraintCode::ZC:
      return "ZC";
    case ConstraintCode::Zy:
      return "Zy";
    case ConstraintCode::p:
      return "p";
    case ConstraintCode::ZQ:
      return "ZQ";
    case ConstraintCode::ZR:
      return "ZR";
    case ConstraintCode::ZS:
      return "ZS";
    case ConstraintCode::ZT:
      return "ZT";
    default:
      llvm_unreachable("Unknown memory constraint");
    }
  }
};

} // end namespace llvm

#endif // LLVM_IR_INLINEASM_H
