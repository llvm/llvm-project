//===- MCExpr.h - Assembly Level Expressions --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCEXPR_H
#define LLVM_MC_MCEXPR_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/SMLoc.h"
#include <cstdint>

namespace llvm {

class MCAsmInfo;
class MCAssembler;
class MCContext;
class MCFixup;
class MCFragment;
class MCSection;
class MCStreamer;
class MCSymbol;
class MCValue;
class raw_ostream;
class StringRef;
class MCSymbolRefExpr;

/// Base class for the full range of assembler expressions which are
/// needed for parsing.
class MCExpr {
public:
  // Allow MC classes to access the private `print` function.
  friend class MCAsmInfo;
  friend class MCFragment;
  friend class MCOperand;
  enum ExprKind : uint8_t {
    Binary,    ///< Binary expressions.
    Constant,  ///< Constant expressions.
    SymbolRef, ///< References to labels and assigned expressions.
    Unary,     ///< Unary expressions.
    Specifier, ///< Expression with a relocation specifier.
    Target     ///< Target specific expression.
  };

private:
  static const unsigned NumSubclassDataBits = 24;
  static_assert(
      NumSubclassDataBits == CHAR_BIT * (sizeof(unsigned) - sizeof(ExprKind)),
      "ExprKind and SubclassData together should take up one word");

  ExprKind Kind;
  /// Field reserved for use by MCExpr subclasses.
  unsigned SubclassData : NumSubclassDataBits;
  SMLoc Loc;

  void print(raw_ostream &OS, const MCAsmInfo *MAI,
             int SurroundingPrec = 0) const;
  bool evaluateAsAbsolute(int64_t &Res, const MCAssembler *Asm,
                          bool InSet) const;

protected:
  using Spec = uint16_t;
  explicit MCExpr(ExprKind Kind, SMLoc Loc, unsigned SubclassData = 0)
      : Kind(Kind), SubclassData(SubclassData), Loc(Loc) {
    assert(SubclassData < (1 << NumSubclassDataBits) &&
           "Subclass data too large");
  }

  LLVM_ABI bool evaluateAsRelocatableImpl(MCValue &Res, const MCAssembler *Asm,
                                          bool InSet) const;

  unsigned getSubclassData() const { return SubclassData; }

public:
  MCExpr(const MCExpr &) = delete;
  MCExpr &operator=(const MCExpr &) = delete;

  /// \name Accessors
  /// @{

  ExprKind getKind() const { return Kind; }
  SMLoc getLoc() const { return Loc; }

  /// @}
  /// \name Utility Methods
  /// @{

  LLVM_ABI void dump() const;

  /// @}
  /// \name Expression Evaluation
  /// @{

  /// Try to evaluate the expression to an absolute value.
  ///
  /// \param Res - The absolute value, if evaluation succeeds.
  /// \return - True on success.
  LLVM_ABI bool evaluateAsAbsolute(int64_t &Res) const;
  LLVM_ABI bool evaluateAsAbsolute(int64_t &Res, const MCAssembler &Asm) const;
  LLVM_ABI bool evaluateAsAbsolute(int64_t &Res, const MCAssembler *Asm) const;

  /// Aggressive variant of evaluateAsRelocatable when relocations are
  /// unavailable (e.g. .fill). Expects callers to handle errors when true is
  /// returned.
  LLVM_ABI bool evaluateKnownAbsolute(int64_t &Res,
                                      const MCAssembler &Asm) const;

  /// Try to evaluate the expression to a relocatable value, i.e. an
  /// expression of the fixed form (a - b + constant).
  ///
  /// \param Res - The relocatable value, if evaluation succeeds.
  /// \param Asm - The assembler object to use for evaluating values.
  /// \return - True on success.
  LLVM_ABI bool evaluateAsRelocatable(MCValue &Res,
                                      const MCAssembler *Asm) const;

  /// Try to evaluate the expression to the form (a - b + constant) where
  /// neither a nor b are variables.
  ///
  /// This is a more aggressive variant of evaluateAsRelocatable. The intended
  /// use is for when relocations are not available, like the .size directive.
  LLVM_ABI bool evaluateAsValue(MCValue &Res, const MCAssembler &Asm) const;

  /// Find the "associated section" for this expression, which is
  /// currently defined as the absolute section for constants, or
  /// otherwise the section associated with the first defined symbol in the
  /// expression.
  LLVM_ABI MCFragment *findAssociatedFragment() const;

  /// @}

  LLVM_ABI static bool evaluateSymbolicAdd(const MCAssembler *, bool,
                                           const MCValue &, const MCValue &,
                                           MCValue &);
};

////  Represent a constant integer expression.
class MCConstantExpr : public MCExpr {
  int64_t Value;

  // Subclass data stores SizeInBytes in bits 0..7 and PrintInHex in bit 8.
  static const unsigned SizeInBytesBits = 8;
  static const unsigned SizeInBytesMask = (1 << SizeInBytesBits) - 1;
  static const unsigned PrintInHexBit = 1 << SizeInBytesBits;

  static unsigned encodeSubclassData(bool PrintInHex, unsigned SizeInBytes) {
    assert(SizeInBytes <= sizeof(int64_t) && "Excessive size");
    return SizeInBytes | (PrintInHex ? PrintInHexBit : 0);
  }

  MCConstantExpr(int64_t Value, bool PrintInHex, unsigned SizeInBytes)
      : MCExpr(MCExpr::Constant, SMLoc(),
               encodeSubclassData(PrintInHex, SizeInBytes)), Value(Value) {}

public:
  /// \name Construction
  /// @{

  LLVM_ABI static const MCConstantExpr *create(int64_t Value, MCContext &Ctx,
                                               bool PrintInHex = false,
                                               unsigned SizeInBytes = 0);

  /// @}
  /// \name Accessors
  /// @{

  int64_t getValue() const { return Value; }
  unsigned getSizeInBytes() const {
    return getSubclassData() & SizeInBytesMask;
  }

  bool useHexFormat() const { return (getSubclassData() & PrintInHexBit) != 0; }

  /// @}

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Constant;
  }
};

///  Represent a reference to a symbol from inside an expression.
///
/// A symbol reference in an expression may be a use of a label, a use of an
/// assembler variable (defined constant), or constitute an implicit definition
/// of the symbol as external.
class MCSymbolRefExpr : public MCExpr {
public:
  // VariantKind isn't ideal for encoding relocation operators because:
  // (a) other expressions, like MCConstantExpr (e.g., 4@l) and MCBinaryExpr
  // (e.g., (a+1)@l), also need it; (b) semantics become unclear (e.g., folding
  // expressions with @). MCSpecifierExpr, as used by AArch64 and RISC-V, offers
  // a cleaner approach.
  enum VariantKind : uint16_t {
    VK_COFF_IMGREL32 = 3, // symbol@imgrel (image-relative)

    FirstTargetSpecifier,
  };

private:
  /// The symbol being referenced.
  const MCSymbol *Symbol;

  explicit MCSymbolRefExpr(const MCSymbol *Symbol, Spec specifier,
                           const MCAsmInfo *MAI, SMLoc Loc = SMLoc());

public:
  /// \name Construction
  /// @{

  static const MCSymbolRefExpr *create(const MCSymbol *Symbol, MCContext &Ctx,
                                       SMLoc Loc = SMLoc()) {
    return MCSymbolRefExpr::create(Symbol, 0, Ctx, Loc);
  }

  LLVM_ABI static const MCSymbolRefExpr *create(const MCSymbol *Symbol,
                                                Spec specifier, MCContext &Ctx,
                                                SMLoc Loc = SMLoc());

  /// @}
  /// \name Accessors
  /// @{

  const MCSymbol &getSymbol() const { return *Symbol; }

  // Some targets encode the relocation specifier within SymA using
  // MCSymbolRefExpr::SubclassData, which is copied to MCValue::Specifier,
  // though this method is now deprecated.
  VariantKind getKind() const { return VariantKind(getSubclassData()); }
  uint16_t getSpecifier() const { return getSubclassData(); }

  /// @}

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::SymbolRef;
  }
};

/// Unary assembler expressions.
class MCUnaryExpr : public MCExpr {
public:
  enum Opcode {
    LNot,  ///< Logical negation.
    Minus, ///< Unary minus.
    Not,   ///< Bitwise negation.
    Plus   ///< Unary plus.
  };

private:
  const MCExpr *Expr;

  MCUnaryExpr(Opcode Op, const MCExpr *Expr, SMLoc Loc)
      : MCExpr(MCExpr::Unary, Loc, Op), Expr(Expr) {}

public:
  /// \name Construction
  /// @{

  LLVM_ABI static const MCUnaryExpr *
  create(Opcode Op, const MCExpr *Expr, MCContext &Ctx, SMLoc Loc = SMLoc());

  static const MCUnaryExpr *createLNot(const MCExpr *Expr, MCContext &Ctx, SMLoc Loc = SMLoc()) {
    return create(LNot, Expr, Ctx, Loc);
  }

  static const MCUnaryExpr *createMinus(const MCExpr *Expr, MCContext &Ctx, SMLoc Loc = SMLoc()) {
    return create(Minus, Expr, Ctx, Loc);
  }

  static const MCUnaryExpr *createNot(const MCExpr *Expr, MCContext &Ctx, SMLoc Loc = SMLoc()) {
    return create(Not, Expr, Ctx, Loc);
  }

  static const MCUnaryExpr *createPlus(const MCExpr *Expr, MCContext &Ctx, SMLoc Loc = SMLoc()) {
    return create(Plus, Expr, Ctx, Loc);
  }

  /// @}
  /// \name Accessors
  /// @{

  /// Get the kind of this unary expression.
  Opcode getOpcode() const { return (Opcode)getSubclassData(); }

  /// Get the child of this unary expression.
  const MCExpr *getSubExpr() const { return Expr; }

  /// @}

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Unary;
  }
};

/// Binary assembler expressions.
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
    OrNot, ///< Bitwise or not.
    Shl,  ///< Shift left.
    AShr, ///< Arithmetic shift right.
    LShr, ///< Logical shift right.
    Sub,  ///< Subtraction.
    Xor   ///< Bitwise exclusive or.
  };

private:
  const MCExpr *LHS, *RHS;

  MCBinaryExpr(Opcode Op, const MCExpr *LHS, const MCExpr *RHS,
               SMLoc Loc = SMLoc())
      : MCExpr(MCExpr::Binary, Loc, Op), LHS(LHS), RHS(RHS) {}

public:
  /// \name Construction
  /// @{

  LLVM_ABI static const MCBinaryExpr *create(Opcode Op, const MCExpr *LHS,
                                             const MCExpr *RHS, MCContext &Ctx,
                                             SMLoc Loc = SMLoc());

  static const MCBinaryExpr *createAdd(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx, SMLoc Loc = SMLoc()) {
    return create(Add, LHS, RHS, Ctx, Loc);
  }

  static const MCBinaryExpr *createAnd(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return create(And, LHS, RHS, Ctx);
  }

  static const MCBinaryExpr *createDiv(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return create(Div, LHS, RHS, Ctx);
  }

  static const MCBinaryExpr *createEQ(const MCExpr *LHS, const MCExpr *RHS,
                                      MCContext &Ctx) {
    return create(EQ, LHS, RHS, Ctx);
  }

  static const MCBinaryExpr *createGT(const MCExpr *LHS, const MCExpr *RHS,
                                      MCContext &Ctx) {
    return create(GT, LHS, RHS, Ctx);
  }

  static const MCBinaryExpr *createGTE(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return create(GTE, LHS, RHS, Ctx);
  }

  static const MCBinaryExpr *createLAnd(const MCExpr *LHS, const MCExpr *RHS,
                                        MCContext &Ctx) {
    return create(LAnd, LHS, RHS, Ctx);
  }

  static const MCBinaryExpr *createLOr(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return create(LOr, LHS, RHS, Ctx);
  }

  static const MCBinaryExpr *createLT(const MCExpr *LHS, const MCExpr *RHS,
                                      MCContext &Ctx) {
    return create(LT, LHS, RHS, Ctx);
  }

  static const MCBinaryExpr *createLTE(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return create(LTE, LHS, RHS, Ctx);
  }

  static const MCBinaryExpr *createMod(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return create(Mod, LHS, RHS, Ctx);
  }

  static const MCBinaryExpr *createMul(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return create(Mul, LHS, RHS, Ctx);
  }

  static const MCBinaryExpr *createNE(const MCExpr *LHS, const MCExpr *RHS,
                                      MCContext &Ctx) {
    return create(NE, LHS, RHS, Ctx);
  }

  static const MCBinaryExpr *createOr(const MCExpr *LHS, const MCExpr *RHS,
                                      MCContext &Ctx) {
    return create(Or, LHS, RHS, Ctx);
  }

  static const MCBinaryExpr *createShl(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return create(Shl, LHS, RHS, Ctx);
  }

  static const MCBinaryExpr *createAShr(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return create(AShr, LHS, RHS, Ctx);
  }

  static const MCBinaryExpr *createLShr(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return create(LShr, LHS, RHS, Ctx);
  }

  static const MCBinaryExpr *createSub(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return create(Sub, LHS, RHS, Ctx);
  }

  static const MCBinaryExpr *createXor(const MCExpr *LHS, const MCExpr *RHS,
                                       MCContext &Ctx) {
    return create(Xor, LHS, RHS, Ctx);
  }

  /// @}
  /// \name Accessors
  /// @{

  /// Get the kind of this binary expression.
  Opcode getOpcode() const { return (Opcode)getSubclassData(); }

  /// Get the left-hand side expression of the binary operator.
  const MCExpr *getLHS() const { return LHS; }

  /// Get the right-hand side expression of the binary operator.
  const MCExpr *getRHS() const { return RHS; }

  /// @}

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Binary;
  }
};

/// Extension point for target-specific MCExpr subclasses to implement.
/// This can encode a relocation operator, serving as a replacement for
/// MCSymbolRefExpr::VariantKind. Ideally, limit this to
/// top-level use, avoiding its inclusion as a subexpression.
///
/// NOTE: All subclasses are required to have trivial destructors because
/// MCExprs are bump pointer allocated and not destructed.
class LLVM_ABI MCTargetExpr : public MCExpr {
  virtual void anchor();

protected:
  MCTargetExpr() : MCExpr(Target, SMLoc()) {}
  virtual ~MCTargetExpr() = default;

public:
  virtual void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const = 0;
  virtual bool evaluateAsRelocatableImpl(MCValue &Res,
                                         const MCAssembler *Asm) const = 0;
  // allow Target Expressions to be checked for equality
  virtual bool isEqualTo(const MCExpr *x) const { return false; }
  // This should be set when assigned expressions are not valid ".set"
  // expressions, e.g. registers, and must be inlined.
  virtual bool inlineAssignedExpr() const { return false; }
  virtual void visitUsedExpr(MCStreamer& Streamer) const = 0;
  virtual MCFragment *findAssociatedFragment() const = 0;

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }
};

/// Extension point for target-specific MCExpr subclasses with a relocation
/// specifier, serving as a replacement for MCSymbolRefExpr::VariantKind.
/// Limit this to top-level use, avoiding its inclusion as a subexpression.
///
/// NOTE: All subclasses are required to have trivial destructors because
/// MCExprs are bump pointer allocated and not destructed.
class LLVM_ABI MCSpecifierExpr : public MCExpr {
protected:
  const MCExpr *Expr;
  // Target-specific relocation specifier code
  const Spec specifier;

  explicit MCSpecifierExpr(const MCExpr *Expr, Spec S, SMLoc Loc = SMLoc())
      : MCExpr(Specifier, Loc), Expr(Expr), specifier(S) {}

public:
  static const MCSpecifierExpr *create(const MCExpr *Expr, Spec S,
                                       MCContext &Ctx, SMLoc Loc = SMLoc());
  static const MCSpecifierExpr *create(const MCSymbol *Sym, Spec S,
                                       MCContext &Ctx, SMLoc Loc = SMLoc());

  Spec getSpecifier() const { return specifier; }
  const MCExpr *getSubExpr() const { return Expr; }

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Specifier;
  }
};

} // end namespace llvm

#endif // LLVM_MC_MCEXPR_H
