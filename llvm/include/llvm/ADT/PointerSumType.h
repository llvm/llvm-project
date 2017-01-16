//===- llvm/ADT/PointerSumType.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_POINTERSUMTYPE_H
#define LLVM_ADT_POINTERSUMTYPE_H

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace llvm {

/// A compile time pair of an integer tag and the pointer-like type which it
/// indexes within a sum type. Also allows the user to specify a particular
/// traits class for pointer types with custom behavior such as over-aligned
/// allocation.
template <uintptr_t N, typename PointerArgT,
          typename TraitsArgT = PointerLikeTypeTraits<PointerArgT>>
struct PointerSumTypeMember {
  enum { Tag = N };
  typedef PointerArgT PointerT;
  typedef TraitsArgT TraitsT;
};

namespace detail {

template <typename TagT, typename... MemberTs>
struct PointerSumTypeHelper;

}

/// A sum type over pointer-like types.
///
/// This is a normal tagged union across pointer-like types that uses the low
/// bits of the pointers to store the tag.
///
/// Each member of the sum type is specified by passing a \c
/// PointerSumTypeMember specialization in the variadic member argument list.
/// This allows the user to control the particular tag value associated with
/// a particular type, use the same type for multiple different tags, and
/// customize the pointer-like traits used for a particular member. Note that
/// these *must* be specializations of \c PointerSumTypeMember, no other type
/// will suffice, even if it provides a compatible interface.
///
/// This type implements all of the comparison operators and even hash table
/// support by comparing the underlying storage of the pointer values. It
/// doesn't support delegating to particular members for comparisons.
///
/// It also default constructs to a zero tag with a null pointer, whatever that
/// would be. This means that the zero value for the tag type is significant
/// and may be desirable to set to a state that is particularly desirable to
/// default construct.
///
/// There is no support for constructing or accessing with a dynamic tag as
/// that would fundamentally violate the type safety provided by the sum type.
template <typename TagT, typename... MemberTs> class PointerSumType {
  uintptr_t Value;

  typedef detail::PointerSumTypeHelper<TagT, MemberTs...> HelperT;

public:
  PointerSumType() : Value(0) {}

  /// A typed constructor for a specific tagged member of the sum type.
  template <TagT N>
  static PointerSumType
  create(typename HelperT::template Lookup<N>::PointerT Pointer) {
    PointerSumType Result;
    void *V = HelperT::template Lookup<N>::TraitsT::getAsVoidPointer(Pointer);
    assert((reinterpret_cast<uintptr_t>(V) & HelperT::TagMask) == 0 &&
           "Pointer is insufficiently aligned to store the discriminant!");
    Result.Value = reinterpret_cast<uintptr_t>(V) | N;
    return Result;
  }

  TagT getTag() const { return static_cast<TagT>(Value & HelperT::TagMask); }

  template <TagT N> bool is() const { return N == getTag(); }

  template <TagT N> typename HelperT::template Lookup<N>::PointerT get() const {
    void *P = is<N>() ? getImpl() : nullptr;
    return HelperT::template Lookup<N>::TraitsT::getFromVoidPointer(P);
  }

  template <TagT N>
  typename HelperT::template Lookup<N>::PointerT cast() const {
    assert(is<N>() && "This instance has a different active member.");
    return HelperT::template Lookup<N>::TraitsT::getFromVoidPointer(getImpl());
  }

  explicit operator bool() const { return Value & HelperT::PointerMask; }
  bool operator==(const PointerSumType &R) const { return Value == R.Value; }
  bool operator!=(const PointerSumType &R) const { return Value != R.Value; }
  bool operator<(const PointerSumType &R) const { return Value < R.Value; }
  bool operator>(const PointerSumType &R) const { return Value > R.Value; }
  bool operator<=(const PointerSumType &R) const { return Value <= R.Value; }
  bool operator>=(const PointerSumType &R) const { return Value >= R.Value; }

  uintptr_t getOpaqueValue() const { return Value; }

protected:
  void *getImpl() const {
    return reinterpret_cast<void *>(Value & HelperT::PointerMask);
  }
};

namespace detail {

/// A helper template for implementing \c PointerSumType. It provides fast
/// compile-time lookup of the member from a particular tag value, along with
/// useful constants and compile time checking infrastructure..
template <typename TagT, typename... MemberTs>
struct PointerSumTypeHelper : MemberTs... {
  // First we use a trick to allow quickly looking up information about
  // a particular member of the sum type. This works because we arranged to
  // have this type derive from all of the member type templates. We can select
  // the matching member for a tag using type deduction during overload
  // resolution.
  template <TagT N, typename PointerT, typename TraitsT>
  static PointerSumTypeMember<N, PointerT, TraitsT>
  LookupOverload(PointerSumTypeMember<N, PointerT, TraitsT> *);
  template <TagT N> static void LookupOverload(...);
  template <TagT N> struct Lookup {
    // Compute a particular member type by resolving the lookup helper ovorload.
    typedef decltype(LookupOverload<N>(
        static_cast<PointerSumTypeHelper *>(nullptr))) MemberT;

    /// The Nth member's pointer type.
    typedef typename MemberT::PointerT PointerT;

    /// The Nth member's traits type.
    typedef typename MemberT::TraitsT TraitsT;
  };

  // Next we need to compute the number of bits available for the discriminant
  // by taking the min of the bits available for each member. Much of this
  // would be amazingly easier with good constexpr support.
  template <uintptr_t V, uintptr_t... Vs>
  struct Min : std::integral_constant<
                   uintptr_t, (V < Min<Vs...>::value ? V : Min<Vs...>::value)> {
  };
  template <uintptr_t V>
  struct Min<V> : std::integral_constant<uintptr_t, V> {};
  enum { NumTagBits = Min<MemberTs::TraitsT::NumLowBitsAvailable...>::value };

  // Also compute the smallest discriminant and various masks for convenience.
  enum : uint64_t {
    MinTag = Min<MemberTs::Tag...>::value,
    PointerMask = static_cast<uint64_t>(-1) << NumTagBits,
    TagMask = ~PointerMask
  };

  // Finally we need a recursive template to do static checks of each
  // member.
  template <typename MemberT, typename... InnerMemberTs>
  struct Checker : Checker<InnerMemberTs...> {
    static_assert(MemberT::Tag < (1 << NumTagBits),
                  "This discriminant value requires too many bits!");
  };
  template <typename MemberT> struct Checker<MemberT> : std::true_type {
    static_assert(MemberT::Tag < (1 << NumTagBits),
                  "This discriminant value requires too many bits!");
  };
  static_assert(Checker<MemberTs...>::value,
                "Each member must pass the checker.");
};

}

// Teach DenseMap how to use PointerSumTypes as keys.
template <typename TagT, typename... MemberTs>
struct DenseMapInfo<PointerSumType<TagT, MemberTs...>> {
  typedef PointerSumType<TagT, MemberTs...> SumType;

  typedef detail::PointerSumTypeHelper<TagT, MemberTs...> HelperT;
  enum { SomeTag = HelperT::MinTag };
  typedef typename HelperT::template Lookup<HelperT::MinTag>::PointerT
      SomePointerT;
  typedef DenseMapInfo<SomePointerT> SomePointerInfo;

  static inline SumType getEmptyKey() {
    return SumType::create<SomeTag>(SomePointerInfo::getEmptyKey());
  }
  static inline SumType getTombstoneKey() {
    return SumType::create<SomeTag>(
        SomePointerInfo::getTombstoneKey());
  }
  static unsigned getHashValue(const SumType &Arg) {
    uintptr_t OpaqueValue = Arg.getOpaqueValue();
    return DenseMapInfo<uintptr_t>::getHashValue(OpaqueValue);
  }
  static bool isEqual(const SumType &LHS, const SumType &RHS) {
    return LHS == RHS;
  }
};

}

#endif
