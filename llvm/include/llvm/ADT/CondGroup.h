//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implements the condition group and associated comparisons.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_CONDGROUP_H
#define LLVM_ADT_CONDGROUP_H

#include "llvm/ADT/ConstexprUtils.h"
#include "llvm/ADT/MetaSet.h"

#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

namespace llvm::cgrp {

/// Marker class used as a base for all CondGroup classes.
///
/// All `CondGroup`-like classes must derive from this base class in order for
/// template logic to function correctly.
struct CondGroupBase {};

namespace cgrp_detail {

template <typename T> using IsCondGroup = std::is_base_of<CondGroupBase, T>;

template <typename T> struct IsStdOptional : std::false_type {};

template <typename T>
struct IsStdOptional<std::optional<T>> : std::true_type {};

template <typename T, bool = std::is_enum_v<T>> struct UnderlyingTypeImpl {
  using type = T;
};

/// Determine whether the specified type is an enum class type.
template <typename T, bool = std::is_enum_v<T>>
class IsEnumClass : public std::false_type {};

template <typename EnumT>
class IsEnumClass<EnumT, true>
    : public std::bool_constant<
          !std::is_convertible_v<EnumT, std::underlying_type_t<EnumT>>> {};

template <typename EnumT>
struct UnderlyingTypeImpl<EnumT, true> : std::underlying_type<EnumT> {};

template <typename T>
using UnderlyingType = typename UnderlyingTypeImpl<T>::type;

/// No integral type could be determined from a type pack.
struct BadIntegerType : CETypeError {};

template <bool AllSigned, bool AllUnsigned, size_t MaxSize>
class GetIntegralTypeImpl {
public:
  using type = BadIntegerType;
};

template <size_t MaxSize> class GetIntegralTypeImpl<true, false, MaxSize> {
  static_assert(MaxSize <= 8u);

  static constexpr auto chooseType() {
    if constexpr (MaxSize == 0u)
      // A zero MaxSize is only possible when the set of values is empty.  Use a
      // simple `int` to satisfy this corner-case.
      return int();

    else if constexpr (MaxSize == 1u)
      return int8_t();

    else if constexpr (MaxSize == 2u)
      return int16_t();

    else if constexpr (MaxSize == 4u)
      return int32_t();

    else
      return int64_t();
  }

public:
  using type = decltype(chooseType());
};

template <size_t MaxSize> class GetIntegralTypeImpl<false, true, MaxSize> {
  static_assert(MaxSize <= 8u);

  static constexpr auto chooseType() {
    if constexpr (MaxSize == 1u)
      return uint8_t();

    else if constexpr (MaxSize == 2u)
      return uint16_t();

    else if constexpr (MaxSize == 4u)
      return uint32_t();

    else
      return uint64_t();
  }

public:
  using type = decltype(chooseType());
};

/// `true` when all the types are unsigned integral or enums with unsigned
/// underlying types; `false` otherwise.
///
/// An empty list of types is considered 'signed'.
template <typename... Tn>
inline constexpr bool AllUnsigned =
    sizeof...(Tn) &&
    std::conjunction_v<std::is_unsigned<UnderlyingType<Tn>>...>;

/// `true` when all the types are signed integral or enums with signed
/// underlying types; `false` otherwise.
///
/// An empty list of types is not considered 'unsigned'.
template <typename... Tn>
inline constexpr bool AllSigned =
    std::conjunction_v<std::is_signed<UnderlyingType<Tn>>...>;

template <typename... Tn>
using GetIntegralType =
    typename GetIntegralTypeImpl<AllSigned<Tn...>, AllUnsigned<Tn...>,
                                 ce_max<size_t>(0, sizeof(Tn)...)>::type;

/// Indicate that multiple distinct enum types were encountered during an
/// `EnumMeet` operation.
struct EnumMeetMultipleEnums : CETypeError {};

template <typename T0, typename T1> class EnumMeetImpl {
  template <typename ArgT>
  static constexpr bool ValidArg =
      std::is_integral_v<ArgT> || std::is_void_v<ArgT> || std::is_enum_v<ArgT>;

  /// Determine the meet type given two non-error types.
  ///
  /// Assume that if only one type is an enum class, then it must be `U0`.
  template <typename U0, typename U1> static constexpr auto chooseTypeLegal() {
    if constexpr (IsEnumClass<U0>::value) {
      if constexpr (!IsEnumClass<U1>::value || std::is_same_v<U0, U1>)
        return U0();
      else
        // Return an error type from this context instead of `static_assert`
        // here in order to provide the instantiating context in the compiler
        // error output.
        return EnumMeetMultipleEnums();
    } else {
      static_assert(
          !IsEnumClass<U1>::value,
          "Assumption that single enum class type occurs in 'U0' violated.");
      // void return type.
      return;
    }
  }

  static constexpr auto chooseType() {
    if constexpr (std::is_base_of_v<CETypeError, T0>) {
      return T0();
    } else if constexpr (std::is_base_of_v<CETypeError, T1>) {
      return T1();
    } else {
      static_assert(ValidArg<T0>,
                    "Unexpected type... expected integral, enum or void type.");
      static_assert(ValidArg<T1>,
                    "Unexpected type... expected integral, enum or void type.");

      // Conditionally swap T0 and T1 before calling `chooseTypeLegal()` to
      // ensure that if only one is an enum class, it occurs as the first type.
      if constexpr (IsEnumClass<T0>::value)
        return chooseTypeLegal<T0, T1>();
      else
        return chooseTypeLegal<T1, T0>();
    }
  }

public:
  using type = decltype(chooseType());
};

template <typename T0, typename T1>
using EnumMeetT = typename EnumMeetImpl<T0, T1>::type;

template <typename... Tn> class GetEnumTypeImpl {};

template <typename T0, typename... Tn>
class GetEnumTypeImpl<T0, Tn...> : public GetEnumTypeImpl<Tn...> {
  using ChildEnumT = typename GetEnumTypeImpl<Tn...>::type;

public:
  using type = EnumMeetT<T0, ChildEnumT>;
};

template <> class GetEnumTypeImpl<> {
public:
  using type = void;
};

template <typename... Tn>
using GetEnumType = typename GetEnumTypeImpl<Tn...>::type;

} // namespace cgrp_detail

/// Recursive tuple implementation.
///
/// \note This exists because it compiles faster on many toolchains compared to
/// using std::tuple.  Certain gcc toolchains seem to blow up with extensive
/// std::tuple use.
template <typename... Tn> class CondGroupTuple {};

namespace cgrp_detail {

template <typename SingleT, typename GroupElemT>
constexpr bool elemEqual(SingleT const &Single, GroupElemT const &Elem) {
  if constexpr (IsCondGroup<GroupElemT>::value) {
    return Elem.equalDisj(Single);
  } else if constexpr (std::is_integral_v<SingleT> &&
                       std::is_enum_v<GroupElemT>) {
    return Single == std::underlying_type_t<GroupElemT>(Elem);
  } else if constexpr (std::is_enum_v<SingleT> &&
                       std::is_integral_v<GroupElemT>) {
    return std::underlying_type_t<SingleT>(Single) == Elem;
  } else if constexpr (std::is_same_v<std::nullopt_t, GroupElemT>) {
    // `std::optional` uses its own operator==() to compare the wrapped
    // optional type to any conditional group type.  That means we may be
    // encountering an optional's wrapped type here.  If we are comparing a
    // non-nullopt_t value, then equality is always `false` because the parent
    // optional contained a valid value.
    return std::is_same_v<std::nullopt_t, SingleT>;
  } else {
    return Single == Elem;
  }
}

} // namespace cgrp_detail

template <typename T0, typename... Tn>
class CondGroupTuple<T0, Tn...> : public CondGroupTuple<Tn...> {
  using RestT = CondGroupTuple<Tn...>;

private:
  T0 CurVal;

public:
  template <typename... Un>
  constexpr CondGroupTuple(T0 const &Cur, Un &&...Rest)
      : RestT(std::forward<Un>(Rest)...), CurVal(Cur) {}

  template <typename... Un>
  constexpr CondGroupTuple(T0 &&Cur, Un &&...Rest)
      : RestT(std::forward<Un>(Rest)...), CurVal(std::move(Cur)) {}

  /// Disjunction of '==' comparisons.
  template <typename U> constexpr bool equalDisj(U const &Single) const {
    return cgrp_detail::elemEqual(Single, CurVal) ||
           getRest().equalDisj(Single);
  }

private:
  constexpr RestT const &getRest() const { return *this; }
};

template <typename T0> class CondGroupTuple<T0> : public CondGroupBase {
  T0 CurVal;

public:
  constexpr CondGroupTuple(T0 const &Cur) : CurVal(Cur) {}
  constexpr CondGroupTuple(T0 &&Cur) : CurVal(std::move(Cur)) {}

  template <typename U> constexpr bool equalDisj(U const &Single) const {
    return cgrp_detail::elemEqual(Single, CurVal);
  }
};

template <> class CondGroupTuple<> : public CondGroupBase {
public:
  template <typename U> bool equalDisj(U const &Single) const { return false; }
};

/// Class which distributes comparisons across all its data.
///
/// The basic `CondGroup` class uses `std::tuple`-like storage for elements of
/// the group.
///
/// This class records a group of conditions which can be compared using
/// the following functions:
///  - `llvm::cgrp::anyOf()`
///
/// Above functions synthesize a comparison class which will distribute
/// equal / inequal comparisons against a single value.  This enables
/// the user to very succinctly specify a comparison across a group of
/// possible values.
///
/// For example:
/// \code{.cpp}
///  enum class Op {
///   Add, AddLo, AddHi,
///   Sub,
///   Mul, MulLo, MulHi,
///   Mad,
///   Load, Store,
///   Barrier,
///  };
///
///  constexpr auto ArithGroup =
///    llvm::cgrp::makeGroup(Op::Add, Op::Sub, Op::Mul);
///
///  if (getOp() == cgrp::anyOf(ArithGroup)) {
///    // Matched at least one of the arithmatic opcodes in `ArithGroup'.
///  }
/// \endcode
///
/// This comparison is functionally eqivalent to the long-form:
/// \code{.cpp}
///  if (getOp() == Op::Add || getOp() == Op::Sub || getOp() == Op::Mul) {
///    // Process Arith op.
///  }
/// \endcode
///
/// The only difference here is that `getOp()` is only called once.
/// Short-circuiting will occur if an early comparison matches (in
/// the case of `cgrp::anyOf()`).
///
/// \note All of the expressions passed to `cgrp::anyOf()` will be evaluated,
/// regardless of short-circuiting due to function call semantics.
///
///
/// Users can nest `CondGroup` objects to compose larger logical groups, and
/// the comparisons will behave efficiently & correctly.
///
/// For example:
/// \code{.cpp}
///  static constexpr auto AddGroup =
///    llvm::cgrp::makeGroup(Op::AddLo, Op::AddHi, Op::Add);
///
///  static constexpr auto MulMadGroup =
///    llvm::cgrp::makeGroup(Op::MulLo, Op::MulHi, Op::Mul, Op::Mad);
///
///  static constexpr auto StrangeGroup =
///    llvm::cgrp::makeGroup(AddGroup, MulMadGroup, Op::Bar);
///
///  if (getOp() == cgrp::anyOf(StrangeGroup))
///    // Opcode belongs to 'StrangeGroup`.
/// \endcode
template <typename... Tn> class CondGroup : public CondGroupTuple<Tn...> {
public:
  using CondGroup::CondGroupTuple::CondGroupTuple;
};

/// `CondGroup`-like class which stores integral or enum elements in a MetaSet
/// (e.g. `MetaBitset` or `MetaSequenceSet`).
///
/// When applicable, this group can be much more space efficient than the tuple
/// representation.  Additionally, the `MetaBitset` can save run-time by
/// querying hundreds of elements in an O(1) bit extraction.
///
/// \tparam MetaSetT  MetaSet type to query for membership.
/// \tparam EnumT  If `void` then all enum or integral types can be compared to
/// the group; otherwise, all queried types must be integral or convertible to
/// this type.
template <typename MetaSetT, typename EnumT = void>
class CondGroupMetaSet : public CondGroupBase,
                         public MetaSetSortedContainer<MetaSetT> {
public:
  using set_type = MetaSetT;

public:
  constexpr CondGroupMetaSet() {}

  /// Disjunction of '==' comparisons.
  template <typename U> constexpr bool equalDisj(U const &Single) const {
    if constexpr (cgrp_detail::IsEnumClass<U>::value) {
      using EnumMeetT = cgrp_detail::EnumMeetT<U, EnumT>;

      static_assert(
          !std::is_base_of_v<CETypeError, EnumMeetT>,
          "enum class type compared with a CondGroupMetaSet containing a "
          "different enum class type.");
    }
    return MetaSetT::contains(Single);
  }
};

/// Create a condition group object containing the specified \p Values.
/// \relates CondGroup
///
/// \param Values  A list of values to forward to the `CondGroup` object.
///
/// \return A `CondGroup` containing all the specified values.
template <typename... Tn> constexpr auto makeGroup(Tn &&...Values) {
  return CondGroup<std::decay_t<Tn>...>(std::forward<Tn>(Values)...);
}
namespace cgrp_detail {

/// Do not create any MetaBitset with more than this number of words (either
/// dense or sparse).
inline constexpr size_t BitsetWordLimit = 8u;

/// Do not create any MetaBitset with more than this number of bits (either
/// dense or sparse).
inline constexpr size_t BitsetBitLimit = BitsetWordLimit * 64u;

template <typename SeqT> class ChooseMetaSetImpl {};

template <typename IntT, IntT... Values>
class ChooseMetaSetImpl<std::integer_sequence<IntT, Values...>> {
  // Don't use a bitset representation if there are too few values since the
  // compiler seems to optimize these quite effectively.
  static constexpr size_t MinValueCount = 4;

  static constexpr auto chooseMetaBitsetType() {
    constexpr IntT MinVal = ce_min<IntT>(Values...);
    constexpr IntT MaxVal = ce_max<IntT>(Values...);

    // Compute the number of words that would be required for a bitset
    // representation.  Do not use a bitset if this exceeds a word limit imposed
    // by the `MetaBitset` class.
    constexpr size_t BitsetNumWords =
        MetaBitsetNumWordsDetailed<IntT, MinVal, MaxVal>;

    // The bitset representation will require this many bytes in .rodata.
    constexpr size_t BitsetArraySize = BitsetNumWords * sizeof(uint64_t);

    constexpr size_t BitsetPreferenceAddend = sizeof...(Values) < 8 ? 16U : 32U;

    constexpr size_t TupleSize = sizeof(IntT) * sizeof...(Values);

    constexpr size_t BitsetArrayUpperBound = ce_min<size_t>(
        // (1.5 * TupleSize) + BitsetPreferenceAddend
        TupleSize + (TupleSize >> 1) + BitsetPreferenceAddend,
        // Max size we want for a single bitset buffer.
        BitsetWordLimit * sizeof(uint64_t));

    // BitsetSize <= 1.5*TupleSize + BitsetPreferenceAddend
    constexpr bool UseSingleMetabitset =
        BitsetArraySize <= BitsetArrayUpperBound;

    if constexpr (UseSingleMetabitset)
      return MakeMetaBitsetDetailed<IntT, MinVal, MaxVal, Values...>();
    else
      return MakeMetaSparseBitset<IntT, BitsetBitLimit, Values...>();
  }

  static constexpr auto chooseMetaSetType() {
    if constexpr (sizeof...(Values) < MinValueCount)
      return MakeMetaSequenceSet<IntT, Values...>();
    else
      return chooseMetaBitsetType();
  }

public:
  using type = decltype(chooseMetaSetType());
};

template <auto... Values> class ChooseGroup {
  using IntT = GetIntegralType<decltype(Values)...>;

  static_assert(
      std::is_integral_v<IntT>,
      "CondGroups of literals must be either integral or enum types and must "
      "all share the same sign.");

  using EnumT = GetEnumType<decltype(Values)...>;

  static_assert(
      !std::is_same_v<cgrp_detail::EnumMeetMultipleEnums, EnumT>,
      "Multiple enum class types encountered while building a CondGroup of "
      "literals; at most one enum class type may be present in the value "
      "pack.");

  static_assert(!std::is_base_of_v<CETypeError, EnumT>,
                "Unspecified error while deriving the enum type for a "
                "CondGroup of literals.");

  using SequenceType =
      std::integer_sequence<IntT, static_cast<IntT>(Values)...>;
  using MetaSetT = typename ChooseMetaSetImpl<SequenceType>::type;

public:
  using type = CondGroupMetaSet<MetaSetT, EnumT>;
};

template <auto... Values> constexpr auto makeLiteralGroup() {
  using GroupT = typename ChooseGroup<Values...>::type;
  return GroupT();
}

} // namespace cgrp_detail

/// Instantiate a condition group of literal values which may be heavily
/// optimized depending on the specified \p Values.
///
/// \tparam Values  Literal values to add to the condition group.
///
/// If the literals meet the following conditions:
///  - All integral or enum types
///  - All the same signedness
///
/// Then compile-time optimizations can be applied to the representation.  The
/// ideal cases for optimizing the representation are:
///
///  1. A single cluster of literals which are close to eachother in value.
///
///  2. Multiple clusters of literals which are close to eachother in value,
///     with a relatively small number of values scattered between the clusters.
///
/// \note The specification of literals does not need to be in any order to
/// recieve representation optimizations.  Additionally, repeat elements in the
/// group are tolerated but discouraged.
///
///
/// \section single_cluster Single Cluster of Values
///
/// If single cluster of values is detected, then a single `MetaBitset` is used
/// to represent the condition group.  This means that the storage will be
/// limited to a `static constexpr` array of 64-bit integers.  Each bit in the
/// array represents a value in the cluster offset from a potentially non-zero
/// starting point.
///
/// The following is an example of a literal group which is represented as a
/// single `MetaBitset`:
/// \code{.cpp}
///  static constexpr auto Fives =
///     cgrp::Literals<5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 10, 20, 30, 40,
///                    50, 60, 70, 80, 90, 100>;
///
///  // Verify the representation.
///  static_assert(std::is_same_v<
///                  std::remove_cv_t<decltype(Fives)>,
///                  CondGroupMetaSet<
///                    MetaBitset<long, 5L,
///                               0x1084210842108421, // Word 0
///                               0x84210842          // Word 1
///                    >
///                  >
///                >);
/// \endcode
///
/// In this example, `Fives` is represented by a single `MetaBitset` with a
/// start offset of `5` and two 64-bit words specified as template parameters.
/// This means that group inclusion queries will be performed in O(1) as a
/// bit-extraction from a 128-bit buffer.
///
///
/// \section multi_cluster Multiple Clusters of Values
///
/// If the specified values span too large of a range to fit in a single
/// `MetaBitset`, then cluster paritioning is performed.  The literals are
/// sorted and then paritioned based on a sliding window of maximal allowed
/// `MetaBitset` size.  Any clusters which are too small (less than a few
/// elements) are added to a fall-back `MetaSequenceSet` (which is the
/// meta-programming equivalent of the tuple representation).  The remaining
/// clusters which are sufficiently large are converted to `MetaBitset`s.
///
/// \code{.cpp}
///  static constexpr auto Clusters =
///      cgrp::Literals<10000, 10002, 10004, 10006, 10008,
///                     1000, 1002, 1004, 1006, 1008,
///                     5000, 8000>;
///
///  static_assert(
///      std::is_same_v<
///          std::remove_cv_t<decltype(Clusters)>,
///          CondGroupMetaSet<
///              MetaBitset<long, 1000, 0x155>,
///              MetaBitset<long, 10000, 0x155>,
///              MetaSequenceSet<std::integer_sequence<long, 5000, 8000>>
///          >
///        >
///      );
/// \endcode
///
/// In the example above, the type infrastructure is able to identify two
/// distinct clusters that are efficiently represented as `MetaBitset`s and two
/// singletons.  One cluster starts at `10,000` and requires one 64-bit word in
/// the `MetaBitset`.  The other cluster starts at `1,000` and also requires a
/// single 64-bit word in its `MetaBitset`.  Finally the two singleton values
/// are tracked in a seperate `MetaSequenceSet` and checked at the end during
/// inclusion queries.
///
///
/// \section singletons Singleton Values
///
/// In the case where no clusters can be formed, the representation is the same
/// as the multi-cluster case except the `MetaBitset`s are omitted (i.e. a
/// single `MetaSequenceSet` is used).
///
/// For example:
/// \code{.cpp}
///     static constexpr auto Singletons =
///         cgrp::Literals<10000, 5000, 8000, 20000, 90000>;
///
///     static_assert(
///         std::is_same_v<
///           std::remove_cv_t<decltype(Singletons)>,
///           CondGroupMetaSet<
///             MetaSequenceSet<
///               std::integer_sequence<
///                 long, 5000, 8000, 10000, 20000, 90000
///               >
///             >
///           >
///         >);
/// \endcode
///
/// No clusters could be identified in the `Singletons` group, so all the values
/// get pushed to a `MetaSequenceSet`.
///
/// \note That the order of the singleton values will always be in ascending
/// order. This is because we already applied a constexpr sort to the original
/// sequence in order to efficiently identify clusters.
///
/// \section restrictions  Literal type restrictions
///
/// Extra type restrictions are applied to literal groups because their values
/// are effectively down-cast to an integral type in order to enable a `MetaSet`
/// representation.  These restrictions are implied in the tuple-like
/// `CondGroup` because each element of the group retains its original type.
///
///
/// **Literal group formation:**
///
/// - Literals may only be integral or enum types.
/// - Integral and enum types may be mixed in a group definition, but at most
///   one enum type may be used.
/// - The signedness of all types must agree (i.e. the signedness of the
///   underlying_type for an enum).
///
/// These group formation restrictions ensure that the group of values can be
/// represented by a `MetaSet` as well as prevent incompatible enum types from
/// inadvertently being cast down to an integral type.
///
/// The following are examples of permitted literal groups:
/// \code{.cpp}
///  enum class Letter : unsigned { A, B, C, D, E, F };
///
///  // All integral values of the same type.
///  static constexpr auto G0 = cgrp::Literals<5, 10, 15>;
///
///  // All integral values, mixed types with same sign.
///  static constexpr auto G1 = cgrp::Literals<5, short(10), long long(15)>;
///
///  // All enum values of the same type.
///  static constexpr auto G2 = cgrp::Literals<Letter::A, Letter::F>;
///
///  // Mixed enum and integral values; same signedness.
///  static constexpr auto G3 = cgrp::Literals<Letter::D, 0u, 1000ull>;
/// \endcode
///
/// The following are examples of illegal literal groups, which will result in a
/// compilation error:
/// \code{.cpp}
///  enum class Letter : unsigned { A, B, C, D, E, F };
///  enum       Number : unsigned { Zero, One, Two, Three };
///
///  // All integral values but mixed signedness.
///  static constexpr auto G0 = cgrp::Literals<1, 2u, 3>;
///
///  // Integral values mixed with enum values differing in signedness of
///  // underlying type.
///  static constexpr auto G1 = cgrp::Literals<One, 2>;
///
///  // Different enum types.
///  static constexpr auto G2 = cgrp::Literals<Zero, Letter::B>;
/// \endcode
///
///
/// \section future_work  Future work
///
/// \todo Apply binary search to elements in a `MetaSequenceSet` when the
/// `size()` is sufficiently large.
///
/// \todo Make literal groups iterable ranges.
///
/// \todo Implement constexpr set operations (`|`, `&`, `-`) for groups of
/// literals which results in an optimized representation of the resulting
/// group.
template <auto... Values>
inline constexpr auto Literals = cgrp_detail::makeLiteralGroup<Values...>();

/// Group which implements disjunctive equivalence queries.
///
/// \tparam GroupT  The type of the group to query for membership.
template <typename GroupT> class AnyOfGroup : public GroupT {
  template <typename T> using IsOptional = cgrp_detail::IsStdOptional<T>;

public:
  using GroupT::GroupT;

  constexpr AnyOfGroup(GroupT const &Group) : GroupT(Group) {}
  constexpr AnyOfGroup(GroupT &&Group) : GroupT(std::move(Group)) {}

  template <typename U, std::enable_if_t<!IsOptional<U>::value, int> = 0>
  friend constexpr bool operator==(U const &Single, AnyOfGroup const &Group) {
    return Group.equalDisj(Single);
  }

  template <typename U, std::enable_if_t<!IsOptional<U>::value, int> = 0>
  friend constexpr bool operator==(AnyOfGroup const &Group, U const &Single) {
    return Group.equalDisj(Single);
  }

  template <typename U, std::enable_if_t<!IsOptional<U>::value, int> = 0>
  friend constexpr bool operator!=(U const &Single, AnyOfGroup const &Group) {
    return !Group.equalDisj(Single);
  }

  template <typename U, std::enable_if_t<!IsOptional<U>::value, int> = 0>
  friend constexpr bool operator!=(AnyOfGroup const &Group, U const &Single) {
    return !Group.equalDisj(Single);
  }
};

/// Create a specialized comparison `CondGroup` object which can be compared
/// against a single value for equality (inclusion) or inequality (exclusion).
/// \relates AnyOfGroup
///
/// \section equality   Equality: Inclusion in parameters
///
/// An equality comparison against the return result of `cgrp::anyOf()` will
/// return `true` if the specified value is equal to any parameters passed to
/// `cgrp::anyOf()`; otherwise it will return `false`.  For example:
///
/// \code{.cpp}
///  assert(  1 == cgrp::anyOf(5, 4, 3, 2, 1));
///  assert( (0 == cgrp::anyOf(5, 4, 3, 2, 1)) == false);
/// \endcode
///
/// An equality comparison of an `cgrp::anyOf()` result logically expands to
/// the following:
/// \code{.cpp}
///  if ( VAL == cgrp::anyOf(A, B, C, D, ...))
///
///    ==>
///
///  auto TMP = VAL;
///  if ( TMP == A || TMP == B || TMP == C || TMP == D || ... )
/// \endcode
///
///
/// \section inequality   Inequality: Exclusion from parameters
///
/// An inequality comparison against the return result of `cgrp::anyOf()` will
/// result in `false` if the specified value is equal to any parameters passed
/// to `cgrp::anyOf()`; otherwise it will return `true`.
///
/// \code{.cpp}
///  assert( (1 != cgrp::anyOf(5, 4, 3, 2, 1)) == false);
///  assert(  0 != cgrp::anyOf(5, 4, 3, 2, 1));
/// \endcode
///
/// An inequality comparison of an `cgrp::anyOf()` result expands to the
/// following:
/// \code{.cpp}
///  if ( VAL != cgrp::anyOf(A, B, C, D, ...))
///
///    ==>
///
///  auto TMP = VAL;
///  if ( TMP != A && TMP != B && TMP != C && TMP != D && ... )
/// \endcode
///
/// Users may also combine simple types with `CondGroup` objects as parameters
/// to `cgrp::anyOf()`. This will still implement short circuiting logic in the
/// order that values appear.  For example:
///
/// \code{.cpp}
///  auto powersOfTwo = llvm::cgrp::makeGroup(1, 2, 4, 8, 16);
///  auto multiplesOfThree = llvm::cgrp::makeGroup(3, 6, 9, 12, 15);
///
///  // 7 does not exist in either group, nor does it match 10 or 11.
///  assert(7 != cgrp::anyOf(powersOfTwo, multiplesOfThree, 10, 11));
/// \endcode
template <typename... Tn> constexpr auto anyOf(Tn &&...Values) {
  using FirstT = std::decay_t<PackElementT<0, Tn...>>;
  if constexpr (sizeof...(Values) == 1 &&
                cgrp_detail::IsCondGroup<FirstT>::value)
    // A single group was passed in, so avoid wrapping it in another
    // CondGroup tuple.
    return AnyOfGroup<FirstT>(std::forward<Tn>(Values)...);
  else
    return AnyOfGroup<CondGroup<std::decay_t<Tn>...>>(
        std::forward<Tn>(Values)...);
}

namespace cgrp_detail {

template <auto... Values> constexpr auto anyOfLiterals() {
  using GroupT = typename cgrp_detail::ChooseGroup<Values...>::type;

  return AnyOfGroup<GroupT>();
}

} // namespace cgrp_detail

/// Instantiate a disjunctive equivalence comparison group which is strongly
/// optimized for the literals specified as template parameters.
///
/// \tparam Values  Literal values to add to the `AnyOf` comparison.
///
/// Functionally equivalent to:
/// \code{.cpp}
///  cgrp::anyOf(cgrp::Literals<Values...>)
/// \endcode
///
/// All of the representation optimizations employed by `cgrp::Literals` are
/// applied, and the instantiated group is wrapped in an `AnyOfGroup`.
///
/// The `AnyOf` group is a specialized comparison `CondGroup` object which can
/// be compared against a single value for equality (inclusion) or inequality
/// (exclusion).
///
///
/// \section equality   Equality: Inclusion in parameters
///
/// An equality comparison against `cgrp::AnyOf<...>` will return `true` if the
/// specified value is equal to any of the template parameters passed to
/// `cgrp::AnyOf<...>`; otherwise it will return `false`.
///
/// For example:
/// \code{.cpp}
///  assert(  1 == cgrp::AnyOf<5, 4, 3, 2, 1>);
///  assert( (0 == cgrp::AnyOf<5, 4, 3, 2, 1>) == false);
/// \endcode
///
/// An equality comparison of an `cgrp::AnyOf<>` logically expands to the
/// following:
/// \code{.cpp}
///  if ( VAL == cgrp::AnyOf<A, B, C, D, ...>)
///
///    ==>
///
///  auto TMP = VAL;
///  if ( TMP == A || TMP == B || TMP == C || TMP == D || ... )
/// \endcode
///
///
/// \section inequality   Inequality: Exclusion from parameters
///
/// An inequality comparison against an `cgrp::AnyOf` instantiation will return
/// in `false` if the specified value is equal to any of the template
/// parameters; otherwise it will return `true`.
///
/// \code{.cpp}
///  assert( (1 != cgrp::AnyOf<5, 4, 3, 2, 1>) == false);
///  assert(  0 != cgrp::AnyOf<5, 4, 3, 2, 1>);
/// \endcode
///
/// An inequality comparison of an `cgrp::AnyOf` logically expands to the
/// following:
/// \code{.cpp}
///  if ( VAL != cgrp::AnyOf<A, B, C, D, ...>)
///
///    ==>
///
///  auto TMP = VAL;
///  if ( TMP != A && TMP != B && TMP != C && TMP != D && ... )
/// \endcode
///
///
/// \section restrictions  Literal type restrictions
///
/// Extra type restrictions are applied to literal groups because their values
/// are effectively down-cast to an integral type in order to enable a `MetaSet`
/// representation.  These restrictions are implied in the tuple-like
/// `CondGroup` because each element of the group retains its original type.
///
///
/// **Literal group formation:**
///
/// - Literals may only be integral or enum types.
/// - Integral and enum types may be mixed in a group definition, but at most
///   one enum type may be used.
/// - The signedness of all types must agree (i.e. the signedness of the
///   underlying_type for an enum).
///
/// These group formation restrictions ensure that the group of values can be
/// represented by a `MetaSet` as well as prevent incompatible enum types from
/// inadvertently being cast down to an integral type.
///
/// The following are examples of permitted literal groups:
/// \code{.cpp}
///  enum class Letter : unsigned { A, B, C, D, E, F };
///
///  // All integral values of the same type.
///  static constexpr auto G0 = cgrp::Literals<5, 10, 15>;
///
///  // All integral values, mixed types with same sign.
///  static constexpr auto G1 = cgrp::Literals<5, short(10), long long(15)>;
///
///  // All enum values of the same type.
///  static constexpr auto G2 = cgrp::Literals<Letter::A, Letter::F>;
///
///  // Mixed enum and integral values; same signedness.
///  static constexpr auto G3 = cgrp::Literals<Letter::D, 0u, 1000ull>;
/// \endcode
///
/// The following are examples of illegal literal groups, which will result in a
/// compilation error:
/// \code{.cpp}
///  enum class Letter : unsigned { A, B, C, D, E, F };
///  enum       Number : unsigned { Zero, One, Two, Three };
///
///  // All integral values but mixed signedness.
///  static constexpr auto G0 = cgrp::Literals<1, 2u, 3>;
///
///  // Integral values mixed with enum values differing in signedness of
///  // underlying type.
///  static constexpr auto G1 = cgrp::Literals<One, 2>;
///
///  // Different enum types.
///  static constexpr auto G2 = cgrp::Literals<Zero, Letter::B>;
/// \endcode
///
///
/// **Literal group comparisons:**
///
/// - The single value compared to the group must be convertible to the
///   underlying integral type used by the `MetaSet` representation.
/// - If the literals included an enum type, then the single value compared to a
///   group must be convertible to that enum type.
///
/// The above comparison restrictions are meant to make the `CondGroup`
/// infrastructure both permissive in the sense that integral values can be
/// comparered to a group derived from an enum class, but not completely
/// circumvent the type system by disallowing comparisons across multiple enum
/// types.
///
/// For example, the following comparison is permitted:
/// \code{.cpp}
///  enum class Letter { A, B, C, D, E, F };
///
///  static_assert(1 == cgrp::AnyOf<Letter::A, Letter::B>);
/// \endcode
///
/// However, the following results in a compilation error:
/// \code{.cpp}
///  enum class Letter { A, B, C, D, E, F };
///  enum Number { Zero, One, Two, Three, Four };
///
///  // Compile error: incomptible enum types.
///  static_assert(Letter::A == cgrp::AnyOf<0, One, 2>);
/// \endcode
template <auto... Values>
inline constexpr auto AnyOf = cgrp_detail::anyOfLiterals<Values...>();

/// Compute the union of two literal condition groups at compile time.
template <typename SetL, typename SetR>
constexpr auto operator|(CondGroupMetaSet<SetL>, CondGroupMetaSet<SetR>) {
  return CondGroupMetaSet<
      MetaSetUnion<SetL, SetR, cgrp_detail::BitsetBitLimit>>();
}

/// Compute the intersection of two literal condition groups at compile time.
template <typename SetL, typename SetR>
constexpr auto operator&(CondGroupMetaSet<SetL>, CondGroupMetaSet<SetR>) {
  return CondGroupMetaSet<
      MetaSetIntersection<SetL, SetR, cgrp_detail::BitsetBitLimit>>();
}

/// Compute the set resulting from subtracting \p SetR from \p SetL at compile
/// time.
template <typename SetL, typename SetR>
constexpr auto operator-(CondGroupMetaSet<SetL>, CondGroupMetaSet<SetR>) {
  return CondGroupMetaSet<
      MetaSetMinus<SetL, SetR, cgrp_detail::BitsetBitLimit>>();
}

} // namespace llvm::cgrp

#endif // LLVM_ADT_CONDGROUP_H
