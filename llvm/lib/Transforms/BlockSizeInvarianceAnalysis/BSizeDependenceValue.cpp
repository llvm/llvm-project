#include "llvm/Transforms/BSizeDependenceValue.h"

#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace llvm;

BSizeDependenceValue BSizeDependenceValue::join(const BSizeDependenceValue& v)
    const {
  if (t_ == BOT) return v;
  if (v.t_ == BOT) return *this;

  // Special case: merge of B_CONST with CONST returns B_CONST
  if (v.t_ == B_CONST && t_ == CONST) return v;
  if (t_ == B_CONST && v.t_ == CONST) return *this;

  if (t_ == v.t_ && isNegative_ == v.isNegative_) {
    if (k_ == v.k_ && !isUnknownK_ && !v.isUnknownK_) return v;
    else {
      auto v = BSizeDependenceValue(t_, isNegative_, nullptr);
      v.isUnknownK_ = true;
      return v;
    }
  } 
  // Merging (b_bsize) with (-b_bsize).
  if (t_ == B_BSIZE && t_ == v.t_ && isNegative_ != v.isNegative_) {
    return BSizeDependenceValue(B_CONST);
  }
  return BSizeDependenceValue(TOP);
}


// Operations
// -----------------

BSizeDependenceValue abstractSum(const BSizeDependenceValue& v1, 
    const BSizeDependenceValue& v2, const Value* res = nullptr) {
  if (v1.t_ == BOT || v2.t_ == BOT) return BSizeDependenceValue(BOT);
  if (v1.t_ == TOP || v2.t_ == TOP) return BSizeDependenceValue(TOP);
  // If both are constants, return (const, res).
  if (v1.t_ == CONST && v2.t_ == CONST) { 
    return BSizeDependenceValue(CONST, false, res);
  }
  // If one of the values is constant and other is (tid) or (bid.bsize),
  // return the other.
  if ((v1.t_ == TID || v1.t_ == BIDBSIZE) && v2.t_ == CONST) { return v1; }
  if ((v2.t_ == TID || v2.t_ == BIDBSIZE) && v1.t_ == CONST) { return v2; }
  // If one of the values is (bid.bsize) and other is (tid) and their multipliers
  // and signs match, return (const, res).
  if (((v1.t_ == TID && v2.t_ == BIDBSIZE) || (v2.t_ == TID && v1.t_ == BIDBSIZE))
       && v1.k_ == v2.k_ && v1.isNegative_ == v2.isNegative_
       && !v1.isUnknownK_ && !v2.isUnknownK_) {
    return BSizeDependenceValue(CONST, false, res);
  }
  // Return (unknown) by default.
  return BSizeDependenceValue(TOP);
}

BSizeDependenceValue abstractProd(const BSizeDependenceValue& v1,
    const BSizeDependenceValue& v2, const Value* res = nullptr) {
  if (v1.t_ == BOT || v2.t_ == BOT) return BSizeDependenceValue(BOT);
  if (v1.t_ == CONST && v2.t_ == CONST) {
    return BSizeDependenceValue(CONST, false, res);
  }
  // Multiplying non-const value with a constant.
  if (v2.t_ == CONST && v2.k_ != nullptr &&
      (v1.t_ != CONST && v1.t_ != TOP) && v1.k_ == nullptr) {
    return BSizeDependenceValue(v1.t_, v1.isNegative_, v2.k_);
  }
  if (v1.t_ == CONST && v1.k_ != nullptr &&
      (v2.t_ != CONST && v2.t_ != TOP) && v2.k_ == nullptr) {
    return BSizeDependenceValue(v2.t_, v2.isNegative_, v1.k_);
  }
  // Multiplying (bid) with (bsize).
  if (((v1.t_ == BID && v2.t_ == BSIZE) || (v2.t_ == BID && v1.t_ == BSIZE)) &&
      (v1.k_ == nullptr || v2.k_ == nullptr)) {
    bool isNegative = (v1.isNegative_ != v2.isNegative_);
    const Value* k = (v1.k_ == nullptr)? v2.k_ : v1.k_;
    return BSizeDependenceValue(BIDBSIZE, isNegative, k);
  }
  // Multiplying (gsize) with (bsize).
  if ((v1.t_ == GSIZE && v2.t_ == BSIZE) ||
      (v2.t_ == GSIZE && v1.t_ == BSIZE)) {
    return BSizeDependenceValue(CONST, false, res);
  } 
  return BSizeDependenceValue(TOP);
}



BSizeDependenceValue abstractNeg(const BSizeDependenceValue& v,
    const Value* res = nullptr) {
  if (v.t_ == BOT) return v;
  if (v.t_ == TOP) return v;
  if (v.t_ == B_CONST) return v;
  if (v.t_ == CONST) return BSizeDependenceValue(CONST, false, res);
  return BSizeDependenceValue(v.t_, !v.isNegative_, v.k_);
}

BSizeDependenceValue operator&&(const BSizeDependenceValue& v1,
    const BSizeDependenceValue& v2) {
  // Precedence for v1. 
  // If already dependent on bsize, return v1, else return v2.
  if (v1.t_ == B_BSIZE) return v1;
  return v2;
}

BSizeDependenceValue operator||(const BSizeDependenceValue& v1,
    const BSizeDependenceValue& v2) {
  // Precedence for v1. 
  // If already dependent on bsize, return v1, else return v2.
  if (v1.t_ == B_BSIZE) return v1;
  return v2;
}

BSizeDependenceValue abstractRel(const BSizeDependenceValue& v1,
    const BSizeDependenceValue& v2) {
  if (v1.t_ == CONST && v2.t_ == CONST) { 
    return BSizeDependenceValue(B_CONST, false);
  }
  return BSizeDependenceValue(B_BSIZE, false);
}

bool operator==(const BSizeDependenceValue& v1,
    const BSizeDependenceValue& v2) {
  return (v1.t_ == v2.t_ && v1.isNegative_ == v2.isNegative_ && 
          ((!v1.isUnknownK_ && !v2.isUnknownK_ && v1.k_ == v2.k_) ||
          (v1.isUnknownK_ && v2.isUnknownK_)));
}

bool operator!=(const BSizeDependenceValue& v1,
    const BSizeDependenceValue& v2) {
  return !(v1 == v2);
}

std::string BSizeDependenceValue::getString() const {
  std::string s;
  if (isAddressType()) s.append("*");
  if (isNegative_) s.append("-");
  if (t_ != CONST && t_ != B_CONST) {
    if (isUnknownK_) { s.append("{ ?? }."); }
    else if (k_)  {
      raw_string_ostream ss(s);
      ss << "{" << *k_ << "}";
      s = ss.str();
      s.append(".");
    }
  }
  switch(t_) {
    case BOT:
      return s.append("u");
    case CONST:
      return s.append("c");
    case B_CONST:
      return s.append("b_c");
    case TID: 
      return s.append("tid");
    case BID: 
      return s.append("bid");
    case BSIZE: 
      return s.append("bdim");
    case B_BSIZE: 
      return s.append("b_bdim");
    case BIDBSIZE: 
      return s.append("bid.bdim");
    case TOP:
    default:
      return s.append("?");
  }
}

bool BSizeDependenceValue::testBSizeDependenceValue() {
  static LLVMContext context;
  const Value* g = new GlobalVariable(
               /*Type */ Type::getInt32Ty(context),
               /*isConstant */ false,
               /*Linkage */ GlobalValue::CommonLinkage,
               /*Initializer=*/0, // has initializer, specified below
               /*Name=*/"g");


  BSizeDependenceValue a = BSizeDependenceValue(TID);
  BSizeDependenceValue b = BSizeDependenceValue(BID);
  BSizeDependenceValue c = BSizeDependenceValue(BSIZE);
  BSizeDependenceValue d = BSizeDependenceValue(CONST, false, g);
  BSizeDependenceValue e;

  errs() << " a : " << a.getString() << "\n";
  errs() << " b : " << b.getString() << "\n";
  errs() << " c : " << c.getString() << "\n";
  errs() << " d : " << d.getString() << "\n";
  errs() << "\n";

  // Test binary operations.
  e = abstractSum(a, b);
  errs() << " e := a + b : " << e.getString() << "\n";
  e = abstractSum(a, d);
  errs() << " e := a + d : " << e.getString() << "\n";
  e = abstractProd(a, b);
  errs() << " e := a * b : " << e.getString() << "\n";
  e = abstractProd(a, d);
  errs() << " e := a * d : " << e.getString() << "\n";
  e = abstractProd(b, c);
  errs() << " e := b * c : " << e.getString() << "\n";
  errs() << "\n";

  // Test assignment, equality.
  errs() << " a == b : " << ((a == b)? "true": "false") << "\n";
  a = b;
  errs() << " a after (a := b) : " << a.getString() << "\n";
  errs() << " a == b : " << ((a == b)? "true": "false") << "\n";
  errs() << "\n";

  // Test join operation.
  e = a.join(b);
  errs() << " e := a.join(b) : " << e.getString() << "\n";
  e = a.join(BSizeDependenceValue(BOT));
  errs() << " e := a.join(bot) : " << e.getString() << "\n";
  e = a.join(BSizeDependenceValue(TOP));
  errs() << " e := a.join(top) : " << e.getString() << "\n";
  errs() << "\n";

  return true;
}
