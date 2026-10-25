#include "llvm/Transforms/MultiplierValue.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace llvm;

// Helper functions: Int to MultiplierValue; MultiplierValue to Int
MultiplierValue MultiplierValue::getMultiplierValue(int x, bool b) {
  if (x == 0) return MultiplierValue(ZERO, b);
  if (x == 1) return MultiplierValue(ONE, b);
  if (x == -1) return MultiplierValue(NEGONE, b);
  return MultiplierValue(TOP);
}

int MultiplierValue::getIntValue() const {
  if(t_ == ZERO) return 0;
  if(t_ == ONE) return 1;
  if(t_ == NEGONE) return -1;
  return 2; // TOP
}

MultiplierValue MultiplierValue::join(const MultiplierValue& v) const {
  if (t_ == BOT) return v;
  if (v.t_ == BOT) return *this;
  if (t_ == v.t_) return v;
  return MultiplierValue(TOP);
}

// Binary Operations

// Integer addition
MultiplierValue operator+(const MultiplierValue& v1, const MultiplierValue& v2) {
  if (v1.t_ == BOT || v2.t_ == BOT) return MultiplierValue(BOT);
  if (v1.t_ == TOP || v2.t_ == TOP) return MultiplierValue(TOP);
  int out = v1.getIntValue() + v2.getIntValue();
  return MultiplierValue::getMultiplierValue(out, false);
}

// Integer multiplication
MultiplierValue operator*(const MultiplierValue& v1, const MultiplierValue& v2) {
  if (v1.t_ == BOT || v2.t_ == BOT) return MultiplierValue(BOT);
  if (v1.t_ == ZERO && v2.t_ == ZERO) return MultiplierValue(ZERO);
  return MultiplierValue(TOP);
}

// Returns abstract value for predicate (v1 == v2);
// If the incoming values are equal, then they have the same dependence on
// thread ID, hence the conditional is thread ID independent. Returns boolean
// value (zero).
// If one incoming value is a constant, and other is linear in thread ID, 
// they can be equal for at most one thread. Returns boolean value (one). 
MultiplierValue eq(const MultiplierValue& v1, const MultiplierValue& v2) {
  if (v1.t_ == BOT || v2.t_ == BOT) return MultiplierValue(BOT);
  if (v1.t_ == v2.t_ && v1.t_ != TOP) return MultiplierValue(ZERO, true);
  if (((v1.t_ == ONE || v1.t_ == NEGONE) && v2.t_ == ZERO) || 
      ((v2.t_ == ONE || v2.t_ == NEGONE) && v1.t_ == ZERO)) {
    return MultiplierValue(ONE, true);
  }
  return MultiplierValue(TOP);
}

// Returns abstract value for predicate (v1 != v2);
// If the incoming values are equal, then they have the same dependence on
// thread ID, hence the conditional is thread ID independent. Returns boolean
// value (zero).
// If one incoming value is a constant, and other is linear in thread ID, 
// they will be unequal for all but one thread. Returns boolean value (negone).
MultiplierValue neq(const MultiplierValue& v1, const MultiplierValue& v2) {
  if (v1.t_ == BOT || v2.t_ == BOT) return MultiplierValue(BOT);
  if (v1.t_ == v2.t_ && v1.t_ != TOP) return MultiplierValue(ZERO, true);
  if (((v1.t_ == ONE || v1.t_ == NEGONE) && v2.t_ == ZERO) ||
      ((v2.t_ == ONE || v2.t_ == NEGONE) && v1.t_ == ZERO)) {
    return MultiplierValue(NEGONE, true);
  }
  return MultiplierValue(TOP);
}

// Returns conjunction of abstract predicate values.
// If both predicates are thread-independent, result is thread-independent. 
// If one of the predicates is true for at most one thread, the conjunction is
// true for at most one thread.
MultiplierValue operator&&(const MultiplierValue& v1, const MultiplierValue& v2) {
  if (v1.t_ == BOT || v2.t_ == BOT) return MultiplierValue(BOT);
  if (v1.t_ == ZERO && v2.t_ == ZERO) return MultiplierValue(ZERO, true);
  if (v1.t_ == ONE || v2.t_ == ONE) return MultiplierValue(ONE, true);
  return MultiplierValue(TOP);
}

// Returns disjunction of abstract predicate values.
// If both predicates are thread-independent, result is thread-independent. 
// If one of the predicates is false for at most one thread, the disjunction is
// false for at most one thread.
MultiplierValue operator||(const MultiplierValue& v1, const MultiplierValue& v2) {
  if (v1.t_ == BOT || v2.t_ == BOT) return MultiplierValue(BOT);
  if (v1.t_ == ZERO && v2.t_ == ZERO) return MultiplierValue(ZERO, true);
  if (v1.t_ == NEGONE || v2.t_ == NEGONE) return MultiplierValue(NEGONE, true);
  return MultiplierValue(TOP);
}

// Negates the abstract value (useful for negation as well.)
MultiplierValue operator-(const MultiplierValue& v) {
  if (v.t_ == BOT) return MultiplierValue(BOT);
  if (v.t_ == TOP) return MultiplierValue(TOP);
  int out = -v.getIntValue();
  return MultiplierValue::getMultiplierValue(out, v.isBool_);
}

bool operator==(const MultiplierValue& v1, const MultiplierValue& v2) {
  return (v1.t_ == v2.t_);
}

bool operator!=(const MultiplierValue& v1, const MultiplierValue& v2) {
  return (v1.t_ != v2.t_);
}

std::string MultiplierValue::getString() const {
  std::string s;
  if (isAddressType()) s.append("*");
  switch(t_) {
    case BOT:
      return s.append("u");
    case ZERO: 
      return s.append("0");
    case ONE: 
      return s.append("1");
    case NEGONE: 
      return s.append("-1");
    case TOP:
    default:
      return s.append(">1");
  }
}

bool MultiplierValue::testMultiplierValue() {
  MultiplierValue a = MultiplierValue(ZERO);
  MultiplierValue b = MultiplierValue(ONE);
  MultiplierValue c;

  errs() << " a : " << a.getString() << "\n";
  errs() << " b : " << b.getString() << "\n";
  errs() << "\n";

  // Test binary operations.
  c = a + b;
  errs() << " c := a + b : " << c.getString() << "\n";
  c = b + c;
  errs() << " c := b + c : " << c.getString() << "\n";
  c = a * b;
  errs() << " c := a * b : " << c.getString() << "\n";
  c = b * c;
  errs() << " c := b * c : " << c.getString() << "\n";
  errs() << "\n";

  // Test relational and boolean operations.
  c = eq(a, b);
  errs() << " eq(a, b) : " << c.getString() << "\n";
  c = neq(a, b);
  errs() << " neq(a, b) : " << c.getString() << "\n";
  c = a && b;
  errs() << " a && b : " << c.getString() << "\n";
  c = a || b;
  errs() << " a || b : " << c.getString() << "\n";

  // Test assignment, equality.
  errs() << " a == b : " << ((a == b)? "true": "false") << "\n";
  a = b;
  errs() << " a after (a := b) : " << a.getString() << "\n";
  errs() << " a == b : " << ((a == b)? "true": "false") << "\n";
  errs() << "\n";

  // Test join operation.
  c = a.join(b);
  errs() << " c := a.join(b) : " << c.getString() << "\n";
  c = a.join(MultiplierValue(BOT));
  errs() << " c := a.join(bot) : " << c.getString() << "\n";
  c = a.join(MultiplierValue(TOP));
  errs() << " c := a.join(top) : " << c.getString() << "\n";
  errs() << "\n";

  return true;
}
