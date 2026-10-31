#ifndef MULTIPLIER_VALUE_H
#define MULTIPLIER_VALUE_H

#include "llvm/Transforms/PointerAbstractValue.h"

// Multiplier Value
// An abstract value used to represent values of integer and boolean variables.
// 
// For integer variables, the abstraction is as follows:
// - (unset) or (bot) represents undefined values.
// - (zero) represents values which are independent of thread ID. Thus,
//     the variable has same value across threads.
// - (one) represents values which have a unit dependence on threadID. Thus the
//     variable has an expression of the form (thread ID + constant).
// - (negone) represents values which a negative unit depdendence on thread ID
//     (-thread ID + constant).
// - (unknown) or (top): other kinds of unknown dependences on thread ID (e.g. 2.threadID,
//     10.threadID + 1).
// For integers it is also refered to as the "thread ID multiplier" (since it
// tracks the multiplier for thread ID).
//
// For booleans variables, (bot) and (top) representations are the same.
// Other values are as follows:
// - (zero) : The variable has same truth value across threads (similar to integers).
// - (one) : The variable is true for at most one thread.
// - (negone) : The variable is false for at most one thread.

namespace uncoalesced_analysis {

enum MultiplierValueType {
  BOT,
  ZERO,
  ONE,
  NEGONE,
  TOP
}; 

}

using namespace uncoalesced_analysis;

class MultiplierValue : public PointerAbstractValue<MultiplierValue> { 
 public:
  MultiplierValue() : t_(MultiplierValueType::BOT), isBool_(false) {}

  MultiplierValue(MultiplierValueType t, bool isBool = false)
      : t_(t), isBool_(isBool) {}

  // Merge values; returns the least value that supersedes both values.
  MultiplierValue join(const MultiplierValue& v) const;

  // Binary operations 
  friend MultiplierValue operator+(const MultiplierValue& v1, const MultiplierValue& v2);
  friend MultiplierValue operator*(const MultiplierValue& v1, const MultiplierValue& v2);
  friend MultiplierValue operator-(const MultiplierValue& v);
  friend MultiplierValue operator&&(const MultiplierValue& v1, const MultiplierValue& v2);
  friend MultiplierValue operator||(const MultiplierValue& v1, const MultiplierValue& v2);
  // Abstract value for predicate (v1 == v2).
  friend MultiplierValue eq(const MultiplierValue& v1, const MultiplierValue& v2);
  // Abstract value for predicate (v1 != v2).
  friend MultiplierValue neq(const MultiplierValue& v1, const MultiplierValue& v2);
  friend bool operator==(const MultiplierValue& v1, const MultiplierValue& v2);
  friend bool operator!=(const MultiplierValue& v1, const MultiplierValue& v2);

  // Getters and setters.
  MultiplierValueType getType() const { return t_; }
  bool isBoolean() const { return isBool_; }

  // Pretty printing 
  std::string getString() const;

  // Test to check correctness.
  static bool testMultiplierValue();

 private:
  // Helper functions.
  int getIntValue() const;
  static MultiplierValue getMultiplierValue(int x, bool b);

  // The type of value.
  MultiplierValueType t_;

  // Is this a boolean value or an integer value?
  bool isBool_;
};

MultiplierValue operator+(const MultiplierValue& v1, const MultiplierValue& v2);
MultiplierValue operator*(const MultiplierValue& v1, const MultiplierValue& v2);
MultiplierValue operator-(const MultiplierValue& v);
MultiplierValue operator&&(const MultiplierValue& v1, const MultiplierValue& v2);
MultiplierValue operator||(const MultiplierValue& v1, const MultiplierValue& v2);
MultiplierValue eq(const MultiplierValue& v1, const MultiplierValue& v2);
MultiplierValue neq(const MultiplierValue& v1, const MultiplierValue& v2);
bool operator==(const MultiplierValue& v1, const MultiplierValue& v2);
bool operator!=(const MultiplierValue& v1, const MultiplierValue& v2);

#endif /* MultiplierValue.h */
