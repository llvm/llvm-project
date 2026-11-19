#ifndef BSIZE_DEPENDENCE_VALUE_H
#define BSIZE_DEPENDENCE_VALUE_H

#include "llvm/Transforms/PointerAbstractValue.h"

#include "llvm/IR/Value.h"

using namespace llvm;

// Block Size Dependence Value
// An abstract value used to track the dependence on block size. Here is the
// abstraction for integer values. Also, the negation of all these values is
// defined as well.
// - (unset) or (bot) represents undefined values.
// - (const) represents values independent of block size.
// - (tid) represents values that are of the form (k.tid + const).
// - (bid) represents values that are of the form (k.bid).
// - (bsize) represents the value (bdim) or (k.bsize).
// - (bidbsize) represents values that are of the form (k.bid.bsize + const).
// - (unknown) or (top): other kinds of values.
//
// For boolean values, we use the following abstraction:
// - (unset) or (bot) represents undefined values.
// - (b_const) represents boolean values independent of block size.
// - (b_bsize) represents the values a function of bid, tid, or bsize.
// - (-b_bsize) represents values that are negation of a bsize-dependent value.
// - (unknown) or (top): other kinds of values.
//
// Additionally, we track if the values are pure constants or a function of
// the global id. The separation is useful for tracking values independent
// of thread and block id altogether.

// TODO:
// Negative values. # done
// Purely constant values.
// Boolean values. # done
// Updating multiplier k_. # done
// Range of values [a, b]?

namespace bsize_invariance_analysis {

enum BSizeDependenceValueType {
  BOT,
  CONST,
  TID,
  BID,
  BSIZE,
  GSIZE,
  BIDBSIZE,
  B_CONST,
  B_BSIZE,
  TOP
}; 

}

using namespace bsize_invariance_analysis;

class BSizeDependenceValue : public PointerAbstractValue<BSizeDependenceValue> { 
 public:
  BSizeDependenceValue() : t_(BSizeDependenceValueType::BOT), isNegative_(false), k_(nullptr),
      isUnknownK_(false), isConstant_(false) {}

  BSizeDependenceValue(BSizeDependenceValueType t, bool isNegative = false,
      const Value* k = nullptr,
      bool isUnknownK = false, bool isConstant = false)
    : t_(t), isNegative_(isNegative), k_(k),
      isUnknownK_(isUnknownK), isConstant_(isConstant) {}
 
  // Merge values; returns the least value that supersedes both values.
  BSizeDependenceValue join(const BSizeDependenceValue& v) const;

  // Binary operations 
  // Parameter 'res' is the resulting LLVM value from the operation; useful for
  // constant updates.
  friend BSizeDependenceValue abstractSum(const BSizeDependenceValue& v1, 
      const BSizeDependenceValue& v2, const Value* res);
  friend BSizeDependenceValue abstractProd(const BSizeDependenceValue& v1,
      const BSizeDependenceValue& v2, const Value* res);
  friend BSizeDependenceValue abstractDiv(const BSizeDependenceValue& v1,
      const BSizeDependenceValue& v2, const Value* res);
  friend BSizeDependenceValue abstractNeg(const BSizeDependenceValue& v,
      const Value* res);
  friend BSizeDependenceValue operator&&(const BSizeDependenceValue& v1,
      const BSizeDependenceValue& v2);
  friend BSizeDependenceValue operator||(const BSizeDependenceValue& v1,
      const BSizeDependenceValue& v2);
  // Abstract value for predicate (v1 op v2), where op is one of "=, !=, <, etc.".
  friend BSizeDependenceValue abstractRel(const BSizeDependenceValue& v1,
      const BSizeDependenceValue& v2);
  friend bool operator==(const BSizeDependenceValue& v1,
      const BSizeDependenceValue& v2);
  friend bool operator!=(const BSizeDependenceValue& v1,
      const BSizeDependenceValue& v2);

  // Getters and setters.
  bool isBoolean() const { return t_ == BSizeDependenceValueType::B_CONST ||
                                  t_ == BSizeDependenceValueType::B_BSIZE; }
  BSizeDependenceValueType getType() const { return t_; }
  bool isNegative() const { return isNegative_; }
  const Value* getMultiplier() const { return k_; }
  bool isConstant() const { return isConstant_; }

  // Pretty printing 
  std::string getString() const;

  // Test to check correctness.
  static bool testBSizeDependenceValue();

 private:
  // The type of value.
  BSizeDependenceValueType t_;

  // Sign of the value.
  bool isNegative_;

  // Value of multiplier k.
  // Note: k_== nullptr implies the multipier is 1.
  const Value* k_;

  // Boolean to check if multiplier is unknown.
  bool isUnknownK_;

  // Is it a pure constant?
  bool isConstant_;
};

BSizeDependenceValue abstractSum(const BSizeDependenceValue& v1, 
    const BSizeDependenceValue& v2, const Value* res);
BSizeDependenceValue abstractProd(const BSizeDependenceValue& v1,
    const BSizeDependenceValue& v2, const Value* res);
BSizeDependenceValue abstractNeg(const BSizeDependenceValue& v,
    const Value* res);
BSizeDependenceValue operator&&(const BSizeDependenceValue& v1,
    const BSizeDependenceValue& v2);
BSizeDependenceValue operator||(const BSizeDependenceValue& v1,
    const BSizeDependenceValue& v2);
BSizeDependenceValue abstractRel(const BSizeDependenceValue& v1,
    const BSizeDependenceValue& v2);
bool operator==(const BSizeDependenceValue& v1,
    const BSizeDependenceValue& v2);
bool operator!=(const BSizeDependenceValue& v1,
    const BSizeDependenceValue& v2);

#endif /* BSizeDependenceValue.h */
