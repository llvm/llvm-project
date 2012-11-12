//===-- llvm/IntegersSubset.h - The subset of integers ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// @file
/// This file contains class that implements constant set of ranges:
/// [<Low0,High0>,...,<LowN,HighN>]. Initially, this class was created for
/// SwitchInst and was used for case value representation that may contain
/// multiple ranges for a single successor.
//
//===----------------------------------------------------------------------===//

#ifndef CONSTANTRANGESSET_H_
#define CONSTANTRANGESSET_H_

#include <list>

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/LLVMContext.h"

namespace llvm {

  // The IntItem is a wrapper for APInt.
  // 1. It determines sign of integer, it allows to use
  //    comparison operators >,<,>=,<=, and as result we got shorter and cleaner
  //    constructions.
  // 2. It helps to implement PR1255 (case ranges) as a series of small patches.
  // 3. Currently we can interpret IntItem both as ConstantInt and as APInt.
  //    It allows to provide SwitchInst methods that works with ConstantInt for
  //    non-updated passes. And it allows to use APInt interface for new methods.
  // 4. IntItem can be easily replaced with APInt.

  // The set of macros that allows to propagate APInt operators to the IntItem.

#define INT_ITEM_DEFINE_COMPARISON(op,func) \
  bool operator op (const APInt& RHS) const { \
    return getAPIntValue().func(RHS); \
  }

#define INT_ITEM_DEFINE_UNARY_OP(op) \
  IntItem operator op () const { \
    APInt res = op(getAPIntValue()); \
    Constant *NewVal = ConstantInt::get(ConstantIntVal->getContext(), res); \
    return IntItem(cast<ConstantInt>(NewVal)); \
  }

#define INT_ITEM_DEFINE_BINARY_OP(op) \
  IntItem operator op (const APInt& RHS) const { \
    APInt res = getAPIntValue() op RHS; \
    Constant *NewVal = ConstantInt::get(ConstantIntVal->getContext(), res); \
    return IntItem(cast<ConstantInt>(NewVal)); \
  }

#define INT_ITEM_DEFINE_ASSIGNMENT_BY_OP(op) \
  IntItem& operator op (const APInt& RHS) {\
    APInt res = getAPIntValue();\
    res op RHS; \
    Constant *NewVal = ConstantInt::get(ConstantIntVal->getContext(), res); \
    ConstantIntVal = cast<ConstantInt>(NewVal); \
    return *this; \
  }

#define INT_ITEM_DEFINE_PREINCDEC(op) \
    IntItem& operator op () { \
      APInt res = getAPIntValue(); \
      op(res); \
      Constant *NewVal = ConstantInt::get(ConstantIntVal->getContext(), res); \
      ConstantIntVal = cast<ConstantInt>(NewVal); \
      return *this; \
    }

#define INT_ITEM_DEFINE_POSTINCDEC(op) \
    IntItem& operator op (int) { \
      APInt res = getAPIntValue();\
      op(res); \
      Constant *NewVal = ConstantInt::get(ConstantIntVal->getContext(), res); \
      OldConstantIntVal = ConstantIntVal; \
      ConstantIntVal = cast<ConstantInt>(NewVal); \
      return IntItem(OldConstantIntVal); \
    }

#define INT_ITEM_DEFINE_OP_STANDARD_INT(RetTy, op, IntTy) \
  RetTy operator op (IntTy RHS) const { \
    return (*this) op APInt(getAPIntValue().getBitWidth(), RHS); \
  }

class IntItem {
  ConstantInt *ConstantIntVal;
  const APInt* APIntVal;
  IntItem(const ConstantInt *V) :
    ConstantIntVal(const_cast<ConstantInt*>(V)),
    APIntVal(&ConstantIntVal->getValue()){}
  const APInt& getAPIntValue() const {
    return *APIntVal;
  }
public:

  IntItem() {}

  operator const APInt&() const {
    return getAPIntValue();
  }

  // Propagate APInt operators.
  // Note, that
  // /,/=,>>,>>= are not implemented in APInt.
  // <<= is implemented for unsigned RHS, but not implemented for APInt RHS.

  INT_ITEM_DEFINE_COMPARISON(<, ult)
  INT_ITEM_DEFINE_COMPARISON(>, ugt)
  INT_ITEM_DEFINE_COMPARISON(<=, ule)
  INT_ITEM_DEFINE_COMPARISON(>=, uge)

  INT_ITEM_DEFINE_COMPARISON(==, eq)
  INT_ITEM_DEFINE_OP_STANDARD_INT(bool,==,uint64_t)

  INT_ITEM_DEFINE_COMPARISON(!=, ne)
  INT_ITEM_DEFINE_OP_STANDARD_INT(bool,!=,uint64_t)

  INT_ITEM_DEFINE_BINARY_OP(*)
  INT_ITEM_DEFINE_BINARY_OP(+)
  INT_ITEM_DEFINE_OP_STANDARD_INT(IntItem,+,uint64_t)
  INT_ITEM_DEFINE_BINARY_OP(-)
  INT_ITEM_DEFINE_OP_STANDARD_INT(IntItem,-,uint64_t)
  INT_ITEM_DEFINE_BINARY_OP(<<)
  INT_ITEM_DEFINE_OP_STANDARD_INT(IntItem,<<,unsigned)
  INT_ITEM_DEFINE_BINARY_OP(&)
  INT_ITEM_DEFINE_BINARY_OP(^)
  INT_ITEM_DEFINE_BINARY_OP(|)

  INT_ITEM_DEFINE_ASSIGNMENT_BY_OP(*=)
  INT_ITEM_DEFINE_ASSIGNMENT_BY_OP(+=)
  INT_ITEM_DEFINE_ASSIGNMENT_BY_OP(-=)
  INT_ITEM_DEFINE_ASSIGNMENT_BY_OP(&=)
  INT_ITEM_DEFINE_ASSIGNMENT_BY_OP(^=)
  INT_ITEM_DEFINE_ASSIGNMENT_BY_OP(|=)

  // Special case for <<=
  IntItem& operator <<= (unsigned RHS) {
    APInt res = getAPIntValue();
    res <<= RHS;
    Constant *NewVal = ConstantInt::get(ConstantIntVal->getContext(), res);
    ConstantIntVal = cast<ConstantInt>(NewVal);
    return *this;
  }

  INT_ITEM_DEFINE_UNARY_OP(-)
  INT_ITEM_DEFINE_UNARY_OP(~)

  INT_ITEM_DEFINE_PREINCDEC(++)
  INT_ITEM_DEFINE_PREINCDEC(--)

  // The set of workarounds, since currently we use ConstantInt implemented
  // integer.

  static IntItem fromConstantInt(const ConstantInt *V) {
    return IntItem(V);
  }
  static IntItem fromType(Type* Ty, const APInt& V) {
    ConstantInt *C = cast<ConstantInt>(ConstantInt::get(Ty, V));
    return fromConstantInt(C);
  }
  static IntItem withImplLikeThis(const IntItem& LikeThis, const APInt& V) {
    ConstantInt *C = cast<ConstantInt>(ConstantInt::get(
        LikeThis.ConstantIntVal->getContext(), V));
    return fromConstantInt(C);
  }
  ConstantInt *toConstantInt() const {
    return ConstantIntVal;
  }
};

template<class IntType>
class IntRange {
protected:
    IntType Low;
    IntType High;
    bool IsEmpty : 1;
    bool IsSingleNumber : 1;

public:
    typedef IntRange<IntType> self;
    typedef std::pair<self, self> SubRes;

    IntRange() : IsEmpty(true) {}
    IntRange(const self &RHS) :
      Low(RHS.Low), High(RHS.High),
      IsEmpty(RHS.IsEmpty), IsSingleNumber(RHS.IsSingleNumber) {}
    IntRange(const IntType &C) :
      Low(C), High(C), IsEmpty(false), IsSingleNumber(true) {}

    IntRange(const IntType &L, const IntType &H) : Low(L), High(H),
      IsEmpty(false), IsSingleNumber(Low == High) {}

    bool isEmpty() const { return IsEmpty; }
    bool isSingleNumber() const { return IsSingleNumber; }

    const IntType& getLow() const {
      assert(!IsEmpty && "Range is empty.");
      return Low;
    }
    const IntType& getHigh() const {
      assert(!IsEmpty && "Range is empty.");
      return High;
    }

    bool operator<(const self &RHS) const {
      assert(!IsEmpty && "Left range is empty.");
      assert(!RHS.IsEmpty && "Right range is empty.");
      if (Low == RHS.Low) {
        if (High > RHS.High)
          return true;
        return false;
      }
      if (Low < RHS.Low)
        return true;
      return false;
    }

    bool operator==(const self &RHS) const {
      assert(!IsEmpty && "Left range is empty.");
      assert(!RHS.IsEmpty && "Right range is empty.");
      return Low == RHS.Low && High == RHS.High;
    }

    bool operator!=(const self &RHS) const {
      return !operator ==(RHS);
    }

    static bool LessBySize(const self &LHS, const self &RHS) {
      return (LHS.High - LHS.Low) < (RHS.High - RHS.Low);
    }

    bool isInRange(const IntType &IntVal) const {
      assert(!IsEmpty && "Range is empty.");
      return IntVal >= Low && IntVal <= High;
    }

    SubRes sub(const self &RHS) const {
      SubRes Res;

      // RHS is either more global and includes this range or
      // if it doesn't intersected with this range.
      if (!isInRange(RHS.Low) && !isInRange(RHS.High)) {

        // If RHS more global (it is enough to check
        // only one border in this case.
        if (RHS.isInRange(Low))
          return std::make_pair(self(Low, High), self());

        return Res;
      }

      if (Low < RHS.Low) {
        Res.first.Low = Low;
        IntType NewHigh = RHS.Low;
        --NewHigh;
        Res.first.High = NewHigh;
      }
      if (High > RHS.High) {
        IntType NewLow = RHS.High;
        ++NewLow;
        Res.second.Low = NewLow;
        Res.second.High = High;
      }
      return Res;
    }
  };

//===----------------------------------------------------------------------===//
/// IntegersSubsetGeneric - class that implements the subset of integers. It
/// consists from ranges and single numbers.
template <class IntTy>
class IntegersSubsetGeneric {
public:
  // Use Chris Lattner idea, that was initially described here:
  // http://lists.cs.uiuc.edu/pipermail/llvm-commits/Week-of-Mon-20120213/136954.html
  // In short, for more compact memory consumption we can store flat
  // numbers collection, and define range as pair of indices.
  // In that case we can safe some memory on 32 bit machines.
  typedef std::vector<IntTy> FlatCollectionTy;
  typedef std::pair<IntTy*, IntTy*> RangeLinkTy;
  typedef std::vector<RangeLinkTy> RangeLinksTy;
  typedef typename RangeLinksTy::const_iterator RangeLinksConstIt;

  typedef IntegersSubsetGeneric<IntTy> self;

protected:

  FlatCollectionTy FlatCollection;
  RangeLinksTy RangeLinks;

  bool IsSingleNumber;
  bool IsSingleNumbersOnly;

public:

  template<class RangesCollectionTy>
  explicit IntegersSubsetGeneric(const RangesCollectionTy& Links) {
    assert(Links.size() && "Empty ranges are not allowed.");

    // In case of big set of single numbers consumes additional RAM space,
    // but allows to avoid additional reallocation.
    FlatCollection.reserve(Links.size() * 2);
    RangeLinks.reserve(Links.size());
    IsSingleNumbersOnly = true;
    for (typename RangesCollectionTy::const_iterator i = Links.begin(),
         e = Links.end(); i != e; ++i) {
      RangeLinkTy RangeLink;
      FlatCollection.push_back(i->getLow());
      RangeLink.first = &FlatCollection.back();
      if (i->getLow() != i->getHigh()) {
        FlatCollection.push_back(i->getHigh());
        IsSingleNumbersOnly = false;
      }
      RangeLink.second = &FlatCollection.back();
      RangeLinks.push_back(RangeLink);
    }
    IsSingleNumber = IsSingleNumbersOnly && RangeLinks.size() == 1;
  }

  IntegersSubsetGeneric(const self& RHS) {
    *this = RHS;
  }

  self& operator=(const self& RHS) {
    FlatCollection.clear();
    RangeLinks.clear();
    FlatCollection.reserve(RHS.RangeLinks.size() * 2);
    RangeLinks.reserve(RHS.RangeLinks.size());
    for (RangeLinksConstIt i = RHS.RangeLinks.begin(), e = RHS.RangeLinks.end();
         i != e; ++i) {
      RangeLinkTy RangeLink;
      FlatCollection.push_back(*(i->first));
      RangeLink.first = &FlatCollection.back();
      if (i->first != i->second)
        FlatCollection.push_back(*(i->second));
      RangeLink.second = &FlatCollection.back();
      RangeLinks.push_back(RangeLink);
    }
    IsSingleNumber = RHS.IsSingleNumber;
    IsSingleNumbersOnly = RHS.IsSingleNumbersOnly;
    return *this;
  }

  typedef IntRange<IntTy> Range;

  /// Checks is the given constant satisfies this case. Returns
  /// true if it equals to one of contained values or belongs to the one of
  /// contained ranges.
  bool isSatisfies(const IntTy &CheckingVal) const {
    if (IsSingleNumber)
      return FlatCollection.front() == CheckingVal;
    if (IsSingleNumbersOnly)
      return std::find(FlatCollection.begin(),
                       FlatCollection.end(),
                       CheckingVal) != FlatCollection.end();

    for (unsigned i = 0, e = getNumItems(); i < e; ++i) {
      if (RangeLinks[i].first == RangeLinks[i].second) {
        if (*RangeLinks[i].first == CheckingVal)
          return true;
      } else if (*RangeLinks[i].first <= CheckingVal &&
                 *RangeLinks[i].second >= CheckingVal)
        return true;
    }
    return false;
  }

  /// Returns set's item with given index.
  Range getItem(unsigned idx) const {
    const RangeLinkTy &Link = RangeLinks[idx];
    if (Link.first != Link.second)
      return Range(*Link.first, *Link.second);
    else
      return Range(*Link.first);
  }

  /// Return number of items (ranges) stored in set.
  unsigned getNumItems() const {
    return RangeLinks.size();
  }

  /// Returns true if whole subset contains single element.
  bool isSingleNumber() const {
    return IsSingleNumber;
  }

  /// Returns true if whole subset contains only single numbers, no ranges.
  bool isSingleNumbersOnly() const {
    return IsSingleNumbersOnly;
  }

  /// Does the same like getItem(idx).isSingleNumber(), but
  /// works faster, since we avoid creation of temporary range object.
  bool isSingleNumber(unsigned idx) const {
    return RangeLinks[idx].first == RangeLinks[idx].second;
  }

  /// Returns set the size, that equals number of all values + sizes of all
  /// ranges.
  /// Ranges set is considered as flat numbers collection.
  /// E.g.: for range [<0>, <1>, <4,8>] the size will 7;
  ///       for range [<0>, <1>, <5>] the size will 3
  unsigned getSize() const {
    APInt sz(((const APInt&)getItem(0).getLow()).getBitWidth(), 0);
    for (unsigned i = 0, e = getNumItems(); i != e; ++i) {
      const APInt Low = getItem(i).getLow();
      const APInt High = getItem(i).getHigh();
      APInt S = High - Low + 1;
      sz += S;
    }
    return sz.getZExtValue();
  }

  /// Allows to access single value even if it belongs to some range.
  /// Ranges set is considered as flat numbers collection.
  /// [<1>, <4,8>] is considered as [1,4,5,6,7,8]
  /// For range [<1>, <4,8>] getSingleValue(3) returns 6.
  APInt getSingleValue(unsigned idx) const {
    APInt sz(((const APInt&)getItem(0).getLow()).getBitWidth(), 0);
    for (unsigned i = 0, e = getNumItems(); i != e; ++i) {
      const APInt Low = getItem(i).getLow();
      const APInt High = getItem(i).getHigh();
      APInt S = High - Low + 1;
      APInt oldSz = sz;
      sz += S;
      if (sz.ugt(idx)) {
        APInt Res = Low;
        APInt Offset(oldSz.getBitWidth(), idx);
        Offset -= oldSz;
        Res += Offset;
        return Res;
      }
    }
    assert(0 && "Index exceeds high border.");
    return sz;
  }

  /// Does the same as getSingleValue, but works only if subset contains
  /// single numbers only.
  const IntTy& getSingleNumber(unsigned idx) const {
    assert(IsSingleNumbersOnly && "This method works properly if subset "
                                  "contains single numbers only.");
    return FlatCollection[idx];
  }
};

//===----------------------------------------------------------------------===//
/// IntegersSubset - currently is extension of IntegersSubsetGeneric
/// that also supports conversion to/from Constant* object.
class IntegersSubset : public IntegersSubsetGeneric<IntItem> {

  typedef IntegersSubsetGeneric<IntItem> ParentTy;

  Constant *Holder;

  static unsigned getNumItemsFromConstant(Constant *C) {
    return cast<ArrayType>(C->getType())->getNumElements();
  }

  static Range getItemFromConstant(Constant *C, unsigned idx) {
    const Constant *CV = C->getAggregateElement(idx);

    unsigned NumEls = cast<VectorType>(CV->getType())->getNumElements();
    switch (NumEls) {
    case 1:
      return Range(IntItem::fromConstantInt(
                     cast<ConstantInt>(CV->getAggregateElement(0U))),
                   IntItem::fromConstantInt(cast<ConstantInt>(
                     cast<ConstantInt>(CV->getAggregateElement(0U)))));
    case 2:
      return Range(IntItem::fromConstantInt(
                     cast<ConstantInt>(CV->getAggregateElement(0U))),
                   IntItem::fromConstantInt(
                   cast<ConstantInt>(CV->getAggregateElement(1))));
    default:
      assert(0 && "Only pairs and single numbers are allowed here.");
      return Range();
    }
  }

  std::vector<Range> rangesFromConstant(Constant *C) {
    unsigned NumItems = getNumItemsFromConstant(C);
    std::vector<Range> r;
    r.reserve(NumItems);
    for (unsigned i = 0, e = NumItems; i != e; ++i)
      r.push_back(getItemFromConstant(C, i));
    return r;
  }

public:

  explicit IntegersSubset(Constant *C) : ParentTy(rangesFromConstant(C)),
                          Holder(C) {}

  IntegersSubset(const IntegersSubset& RHS) :
    ParentTy(*(const ParentTy *)&RHS), // FIXME: tweak for msvc.
    Holder(RHS.Holder) {}

  template<class RangesCollectionTy>
  explicit IntegersSubset(const RangesCollectionTy& Src) : ParentTy(Src) {
    std::vector<Constant*> Elts;
    Elts.reserve(Src.size());
    for (typename RangesCollectionTy::const_iterator i = Src.begin(),
         e = Src.end(); i != e; ++i) {
      const Range &R = *i;
      std::vector<Constant*> r;
      if (R.isSingleNumber()) {
        r.reserve(2);
        // FIXME: Since currently we have ConstantInt based numbers
        // use hack-conversion of IntItem to ConstantInt
        r.push_back(R.getLow().toConstantInt());
        r.push_back(R.getHigh().toConstantInt());
      } else {
        r.reserve(1);
        r.push_back(R.getLow().toConstantInt());
      }
      Constant *CV = ConstantVector::get(r);
      Elts.push_back(CV);
    }
    ArrayType *ArrTy =
        ArrayType::get(Elts.front()->getType(), (uint64_t)Elts.size());
    Holder = ConstantArray::get(ArrTy, Elts);
  }

  operator Constant*() { return Holder; }
  operator const Constant*() const { return Holder; }
  Constant *operator->() { return Holder; }
  const Constant *operator->() const { return Holder; }
};

}

#endif /* CONSTANTRANGESSET_H_ */
