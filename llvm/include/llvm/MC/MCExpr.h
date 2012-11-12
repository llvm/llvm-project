//===- MCExpr.h - Assembly Level Expressions --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCEXPR_H
#define LLVM_MC_MCEXPR_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
class MCAsmLayout;
class MCAssembler;
class MCContext;
class MCSection;
class MCSectionData;
class MCSymbol;
class MCValue;
class raw_ostream;
class StringRef;
typedef DenseMap<const MCSectionData*, uint64_t> SectionAddrMap;

/// MCExpr - Base class for the full range of assembler expressions which are
/// needed for parsing.
class MCExpr {
public:
  enum ExprKind {
    Binary,    ///< Binary expressions.
    Constant,  ///< Constant expressions.
    SymbolRef, ///< References to labels and assigned expressions.
    Unary,     ///< Unary expressions.
    Target     ///< Target specific expression.
  };

private:
  ExprKind Kind;

  MCExpr(const MCExpr&) LLVM_DELETED_FUNCTION;
  void operator=(const MCExpr&) LLVM_DELETED_FUNCTION;

  bool EvaluateAsAbsolute(int64_t &Res, const MCAssembler *Asm,
                          const MCAsmLayout *Layout,
                          const SectionAddrMap *Addrs) const;
protected:
  explicit MCExpr(ExprKind _Kind) : Kind(_Kind) {}

  bool EvaluateAsRelocatableImpl(MCValue &Res, const MCAssembler *Asm,
                                 const MCAsmLayout *Layout,
                                 const SectionAddrMap *Addrs,
                                 bool InSet) const;
public:
  /// @name Accessors
  /// @{

  ExprKind getKind() const { return Kind; }

  /// @}
  /// @name Utility Methods
  /// @{

  void print(raw_ostream &OS) const;
  void dump() const;

  /// @}
  /// @name Expression Evaluation
  /// @{

  /// EvaluateAsAbsolute - Try to evaluate the expression to an absolute value.
  ///
  /// @param Res - The absolute value, if evaluation succeeds.
  /// @param Layout - The assembler layout object to use for evaluating symbol
  /// values. If not given, then only non-symbolic expressions will be
  /// evaluated.
  /// @result - True on success.
  bool EvaluateAsAbsolute(int64_t &Res, const MCAsmLayout &Layout,
                          const SectionAddrMap &Addrs) const;
  bool EvaluateAsAbsolute(int64_t &Res) const;
  bool EvaluateAsAbsolute(int64_t &Res, const MCAssembler &Asm) const;
  bool EvaluateAsAbsolute(int64_t &Res, const MCAsmLayout &Layout) const;

  /// EvaluateAsRelocatable - Try to evaluate the expression to a relocatable
  /// value, i.e. an expression of the fixed form (a - b + constant).
  ///
  /// @param Res - The relocatable value, if evaluation succeeds.
  /// @param Layout - The assembler layout object to use for evaluating values.
  /// @result - True on success.
  bool EvaluateAsRelocatable(MCValue &Res, const MCAsmLayout &Layout) const;

  /// FindAssociatedSection - Find the "associated section" for this expression,
  /// which is currently defined as the absolute section for constants, or
  /// otherwise the section associated with the first defined symbol in the
  /// expression.
  const MCSection *FindAssociatedSection() const;

  /// @}
};

inline raw_ostream &operator<<(raw_ostream &OS, const MCExpr &E) {
  E.print(OS);
  return OS;
}

//// MCConstantExpr - Represent a constant integer expression.
class MCConstantExpr : public MCExpr {
  int64_t Value;

  explicit MCConstantExpr(int64_t _Value)
    : MCExpr(MCExpr::Constant), Value(_Value) {}

public:
  /// @name Construction
  /// @{

  static const MCConstantExpr *Create(int64_t Value, MCContext &Ctx);

  /// @}
  /// @name Accessors
  /// @{

  int64_t getValue() const { return Value; }

  /// @}

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Constant;
  }
};

/// MCSymbolRefExpr - Represent a reference to a symbol from inside an
/// expression.
///
/// A symbol reference in an expression may be a use of a label, a use of an
/// assembler variable (defined constant), or constitute an implicit definition
/// of the symbol as external.
class MCSymbolRefExpr : public MCExpr {
public:
  enum VariantKind {
    VK_None,
    VK_Invalid,

    VK_GOT,
    VK_GOTOFF,
    VK_GOTPCREL,
    VK_GOTTPOFF,
    VK_INDNTPOFF,
    VK_NTPOFF,
    VK_GOTNTPOFF,
    VK_PLT,
    VK_TLSGD,
    VK_TLSLD,
    VK_TLSLDM,
    VK_TPOFF,
    VK_DTPOFF,
    VK_TLVP,      // Mach-O thread local variable relocation
    VK_SECREL,
    // FIXME: We'd really like to use the generic Kinds listed above for these.
    VK_ARM_PLT,   // ARM-style PLT references. i.e., (PLT) instead of @PLT
    VK_ARM_TLSGD, //   ditto for TLSGD, GOT, GOTOFF, TPOFF and GOTTPOFF
    VK_ARM_GOT,
    VK_ARM_GOTOFF,
    VK_ARM_TPOFF,
    VK_ARM_GOTTPOFF,
    VK_ARM_TARGET1,
    VK_ARM_TARGET2,

    VK_PPC_TOC,          // TOC base
    VK_PPC_TOC_ENTRY,    // TOC entry
    VK_PPC_DARWIN_HA16,  // ha16(symbol)
    VK_PPC_DARWIN_LO16,  // lo16(symbol)
    VK_PPC_GAS_HA16,     // symbol@ha
    VK_PPC_GAS_LO16,      // symbol@l
    VK_PPC_TPREL16_HA,   // symbol@tprel@ha
    VK_PPC_TPREL16_LO,   // symbol@tprel@l

    VK_Mips_GPREL,
    VK_Mips_GOT_CALL,
    VK_Mips_GOT16,
    VK_Mips_GOT,
    VK_Mips_ABS_HI,
    VK_Mips_ABS_LO,
    VK_Mips_TLSGD,
    VK_Mips_TLSLDM,
    VK_Mips_DTPREL_HI,
    VK_Mips_DTPREL_LO,
    VK_Mips_GOTTPREL,
    VK_Mips_TPREL_HI,
    VK_Mips_TPREL_LO,
    VK_Mips_GPOFF_HI,
    VK_Mips_GPOFF_LO,
    VK_Mips_GOT_DISP,
    VK_Mips_GOT_PAGE,
    VK_Mips_GOT_OFST,
    VK_Mips_HIGHER,
    VK_Mips_HIGHEST
  };

private:
  /// The symbol being referenced.
  const MCSymbol *Symbol;

  /// The symbol reference modifier.
  const VariantKind Kind;

  explicit MCSymbolRefExpr(const MCSymbol *_Symbol, VariantKind _Kind)
    : MCExpr(MCExpr::SymbolRef), Symbol(_Symbol), Kind(_Kind) {
    assert(Symbol);
  }

public:
  /// @name Construction
  /// @{

  static const MCSymbolRefExpr *Create(const MCSymbol *Symbol, MCContext &Ctx) {
    return MCSymbolRefExpr::Create(Symbol, VK_None, Ctx);
  }

  static const MCSymbolRefExpr *Create(const MCSymbol *Symbol, VariantKind Kind,
                                       MCContext &Ctx);
  static const MCSymbolRefExpr *Create(StringRef Name, VariantKind Kind,
                                       MCContext &Ctx);

  /// @}
  /// @name Accessors
  /// @{

  const MCSymbol &getSymbol() const { return *Symbol; }

  VariantKind getKind() const { return Kind; }

  /// @}
  /// @name Static Utility Functions
  /// @{

  static StringRef getVariantKindName(VariantKind Kind);

  static VariantKind getVariantKindForName(StringRef Name);

  /// @}

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::SymbolRef;
  }
};

/// MCUnaryExpr - Unary assembler expressions.
class MCUnaryExpr : public MCExpr {
public:
  enum Opcode {
    LNot,  ///< Logical negation.
    Minus, ///< Unary minus.
    Not,   ///< Bitwise negation.
    Plus   ///< Unary plus.
  };

private:
  Opcode Op;
  const MCExpr *Expr;

  MCUnaryExpr(Opcode _Op, const MCExpr *_Expr)
    : MCExpr(MCExpr::Unary), Op(_Op), Expr(_Expr) {}

public:
  /// @name Construction
  /// @{

  static const MCUnaryExpr *Create(Opcode Op, const MCExpr *Expr,
                                   MCContext &Ctx);
  static const MCUnaryExpr *CreateLNot(const MCExpr *Expr, MCContext &Ctx) {
    return Create(LNot, Expr, Ctx);
  }
  static const MCUnaryExpr *CreateMinus(const MCExpr *Expr, MCContext &Ctx) {
    return Create(Minus, Expr, Ctx);
  }
  static const MCUnaryExpr *CreateNot(const MCExpr *Expr, MCContext &Ctx) {
    return Create(Not, Expr, Ctx);
  }
  static const MCUnaryExpr *CreatePlus(const MCExpr *Expr, MCContext &Ctx) {
    return Create(Plus, Expr, Ctx);
  }

  /// @}
  /// @name Accessors
  /// @{

  /// getOpcode - Get the kind of this unary expression.
  Opcode getOpcode() const { return Op; }

  /// getSubExpr - Get the child of this unary expression.
  const MCExpr *getSubExpr() const { return Expr; }

  /// @}

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Unary;
  }
};

/// MCBinaryExpr - Binary assembler expressions.
class MCBinaryExpr : public MCExpr {
public:
  enum Opcode {
    Add,  ///< Addition.
    And,  ///< Bitwise and.
    Div,  ///< Signed division.
    EQ,   ///< Equality comparison.
    GT,   ///< Signed greater than comparison (result is either 0 or some
          ///< target-specific non-zero value)
    GTE,  ///< Signed greater than or equal comparison (result is either 0 or
          ///< some target-specific non-zero value).
    LAnd, ///< Logical and.
    LOr,  ///< Logical or.
    LT,   ///< Signed less than comparison (result is either 0 or
          ///< some target-specific non-zero value).
    LTE,  ///< Signed less than or equal comparison (result is either 0 or
          ///< some target-specific non-zero value).
    Mod,  ///< Signed remainder.
    Mul,  ///< Multiplication.
    NE,   ///< Inequality comparison.
    Or,   ///< Bitwise or.
    Shl,  ///< Shift left.
    Shr,  ///< Shift right (arithmetic or logical, depending on target)
    Sub,  ///< Subtraction.
    Xor   ///< Bitwise exclusive or.
  };

private:
  Opcode Op;
  const MCExpr *LHS, *RHS;

  MCBinaryExpr(Opcode _Op, const MCExpr *_LHS, const MCExpr *_RHS)
    : MCExpr(MCExpr::Binary), Op(_Op), LHS(_LHS), RHS(_RHS) {}

public:
  /// @name Construction
  /// @{

  static const MCBinaryExpr *Create(Opcode Op, const MCExpr *LHS,
                                    const MCExpr *RHS, MCContext &Ctx);
  static const MCBinaryExpr *CreateAdd(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return Create(Add, LHS, RHS, Ctx);
  }
  static const MCBinaryExpr *CreateAnd(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return Create(And, LHS, RHS, Ctx);
  }
  static const MCBinaryExpr *CreateDiv(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return Create(Div, LHS, RHS, Ctx);
  }
  static const MCBinaryExpr *CreateEQ(const MCExpr *LHS, const MCExpr *RHS,
                                      MCContext &Ctx) {
    return Create(EQ, LHS, RHS, Ctx);
  }
  static const MCBinaryExpr *CreateGT(const MCExpr *LHS, const MCExpr *RHS,
                                      MCContext &Ctx) {
    return Create(GT, LHS, RHS, Ctx);
  }
  static const MCBinaryExpr *CreateGTE(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return Create(GTE, LHS, RHS, Ctx);
  }
  static const MCBinaryExpr *CreateLAnd(const MCExpr *LHS, const MCExpr *RHS,
                                        MCContext &Ctx) {
    return Create(LAnd, LHS, RHS, Ctx);
  }
  static const MCBinaryExpr *CreateLOr(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return Create(LOr, LHS, RHS, Ctx);
  }
  static const MCBinaryExpr *CreateLT(const MCExpr *LHS, const MCExpr *RHS,
                                      MCContext &Ctx) {
    return Create(LT, LHS, RHS, Ctx);
  }
  static const MCBinaryExpr *CreateLTE(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return Create(LTE, LHS, RHS, Ctx);
  }
  static const MCBinaryExpr *CreateMod(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return Create(Mod, LHS, RHS, Ctx);
  }
  static const MCBinaryExpr *CreateMul(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return Create(Mul, LHS, RHS, Ctx);
  }
  static const MCBinaryExpr *CreateNE(const MCExpr *LHS, const MCExpr *RHS,
                                      MCContext &Ctx) {
    return Create(NE, LHS, RHS, Ctx);
  }
  static const MCBinaryExpr *CreateOr(const MCExpr *LHS, const MCExpr *RHS,
                                      MCContext &Ctx) {
    return Create(Or, LHS, RHS, Ctx);
  }
  static const MCBinaryExpr *CreateShl(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return Create(Shl, LHS, RHS, Ctx);
  }
  static const MCBinaryExpr *CreateShr(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return Create(Shr, LHS, RHS, Ctx);
  }
  static const MCBinaryExpr *CreateSub(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return Create(Sub, LHS, RHS, Ctx);
  }
  static const MCBinaryExpr *CreateXor(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return Create(Xor, LHS, RHS, Ctx);
  }

  /// @}
  /// @name Accessors
  /// @{

  /// getOpcode - Get the kind of this binary expression.
  Opcode getOpcode() const { return Op; }

  /// getLHS - Get the left-hand side expression of the binary operator.
  const MCExpr *getLHS() const { return LHS; }

  /// getRHS - Get the right-hand side expression of the binary operator.
  const MCExpr *getRHS() const { return RHS; }

  /// @}

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Binary;
  }
};

/// MCTargetExpr - This is an extension point for target-specific MCExpr
/// subclasses to implement.
///
/// NOTE: All subclasses are required to have trivial destructors because
/// MCExprs are bump pointer allocated and not destructed.
class MCTargetExpr : public MCExpr {
  virtual void anchor();
protected:
  MCTargetExpr() : MCExpr(Target) {}
  virtual ~MCTargetExpr() {}
public:

  virtual void PrintImpl(raw_ostream &OS) const = 0;
  virtual bool EvaluateAsRelocatableImpl(MCValue &Res,
                                         const MCAsmLayout *Layout) const = 0;
  virtual void AddValueSymbols(MCAssembler *) const = 0;
  virtual const MCSection *FindAssociatedSection() const = 0;

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }
};

} // end namespace llvm

#endif
