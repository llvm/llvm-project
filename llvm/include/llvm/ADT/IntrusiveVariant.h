//===- IntrusiveVariant.h - Compact type safe union -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides IntrusiveVariant, a class template modeled in the spirit
// of std::variant, but leveraging the "common initial sequence" rule for union
// members to store the runtime tag at the beginning of the IntrusiveVariant's
// alternative types, allowing for it to be packed more efficiently into bits
// that would otherwise be used for padding.
//
// However, this requires several restrictions be placed on valid alternative
// types. All alternative types of an IntrusiveVariant must:
//
//  * Be standard-layout. This implies (among other things):
//    * All non-static data members must have the same access control.
//    * All non-static data members must be declared in only one class in the
//      inheritence hierarchy.
//    * No virtual methods.
//  * Begin their class definition by invoking the
//    DECLARE_INTRUSIVE_ALTERNATIVE macro. This declares a member named
//    `IntrusiveVariantTagMember` which must not be referenced outside of the
//    implementation of IntrusiveVariant, and declares some `friend` types to
//    make the tag accessible to the implementation.
//
// Additionally, some features were omitted that are present in the C++17
// std::variant to keep the code simpler:
//
//  * All alternative types must be trivially-destructible.
//  * All copy/move constructors and assignment operators for the variant are
//    disabled if any type is not trivially-constructible and/or
//    trivially-copyable, respectively.
//  * All alternative types must be unique, and cannot be referred to by index.
//  * No equivalent to std::monostate. An instantiation must have at least
//    IntrusiveVariant::MinNumberOfAlternatives alternatives.
//
// If a use case for the above materializes these can always be added
// retroactively.
//
// Example:
//
//  class AltInt {
//    DECLARE_INTRUSIVE_ALTERNATIVE
//    int Int;
//
//  public:
//    AltInt() : Int(0) {}
//    AltInt(int Int) : Int(Int) {}
//    int getInt() const { return Int; }
//    void setInt(int Int) { this->Int = Int; }
//  };
//
//  class AltDouble {
//    DECLARE_INTRUSIVE_ALTERNATIVE
//    double Double;
//
//  public:
//    AltDouble(double Double) : Double(Double) {}
//    double getDouble() const { return Double; }
//    void setDouble(double Double) { this->Double = Double; }
//  };
//
//  class AltComplexInt {
//    DECLARE_INTRUSIVE_ALTERNATIVE
//    int Real;
//    int Imag;
//
//  public:
//    AltComplexInt(int Real, int Imag) : Real(Real), Imag(Imag) {}
//    int getReal() const { return Real; }
//    void setReal(int Real) { this->Real = Real; }
//    int getImag() const { return Imag; }
//    void setImag(int Imag) { this->Imag = Imag; }
//  };
//
//  TEST(VariantTest, HeaderExample) {
//    using MyVariant = IntrusiveVariant<AltInt, AltDouble, AltComplexInt>;
//
//    MyVariant DefaultConstructedVariant;
//    ASSERT_TRUE(DefaultConstructedVariant.holdsAlternative<AltInt>());
//    ASSERT_EQ(DefaultConstructedVariant.get<AltInt>().getInt(), 0);
//    MyVariant Variant{in_place_type<AltComplexInt>, 4, 2};
//    ASSERT_TRUE(Variant.holdsAlternative<AltComplexInt>());
//    int NonSense = visit(
//        makeVisitor(
//            [](AltInt &AI) { return AI.getInt(); },
//            [](AltDouble &AD) { return static_cast<int>(AD.getDouble()); },
//            [](AltComplexInt &ACI) { return ACI.getReal() + ACI.getImag(); }),
//        Variant);
//    ASSERT_EQ(NonSense, 6);
//    Variant.emplace<AltDouble>(2.0);
//    ASSERT_TRUE(Variant.holdsAlternative<AltDouble>());
//    Variant.get<AltDouble>().setDouble(3.0);
//    AltDouble AD = Variant.get<AltDouble>();
//    double D = AD.getDouble();
//    ASSERT_EQ(D, 3.0);
//    Variant.emplace<AltComplexInt>(4, 5);
//    ASSERT_EQ(Variant.get<AltComplexInt>().getReal(), 4);
//    ASSERT_EQ(Variant.get<AltComplexInt>().getImag(), 5);
//  }
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_INTRUSIVEVARIANT_H
#define LLVM_ADT_INTRUSIVEVARIANT_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/VariantTraits.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <type_traits>
#include <utility>

namespace llvm {

template <typename... Ts> class IntrusiveVariant;

/// Helper to get the number of alternative types of a (possibly cv-qualified)
/// IntrusiveVariant type as a constexpr. See std::variant_size.
template <typename T>
struct IntrusiveVariantSize : IntrusiveVariantSize<std::remove_cv_t<T>> {};
template <typename... Ts>
struct IntrusiveVariantSize<IntrusiveVariant<Ts...>>
    : std::integral_constant<size_t, sizeof...(Ts)> {};

/// Simple value type which must be the first member of all alternative types
/// of an IntrusiveVariant. See DECLARE_INTRUSIVE_ALTERNATIVE.
///
/// The internal implementation assumes this is layout-compatible with the
/// "common initial sequence" of all alternative types contained in the private
/// union of the IntrusiveVariant.
struct IntrusiveVariantTag {
  uint8_t Index = std::numeric_limits<uint8_t>::max();
  IntrusiveVariantTag() {}
  IntrusiveVariantTag(uint8_t Index) : Index(Index) {}
};

/// A helper macro to add the declarations needed to use a type as an
/// alternative for IntrusiveVariant. Must be the first declaration of the
/// class.
#define DECLARE_INTRUSIVE_ALTERNATIVE                                          \
  ::llvm::IntrusiveVariantTag IntrusiveVariantTagMember;                       \
  template <typename...> friend class ::llvm::IntrusiveVariant;                \
  template <size_t, typename, typename...>                                     \
  friend union ::llvm::detail::UnionImpl;

namespace detail {
// This struct is used to access the intrusive tag of the alternative types.
//
// All such types must be have an initial sequence which is layout-compatible
// with this struct or the access causes undefined behavior.
struct CommonInitialSequenceT {
  IntrusiveVariantTag Tag;
};

// The inner implementation of the "type safe union". Members are only
// accessible directly via an Index, so IntrusiveVariant must use indexOf to
// convert a pair of T and Ts... into an index.
//
// Effectively implemented as a "linked list" of recursively defined union
// templates. This is the recursive portion of the definition.
//
// We use in_place_index_t here both to disambiguate the constructor and to make
// defining the overload set for getMember more natural.
template <size_t Index, typename HeadT, typename... TailTs> union UnionImpl {
  using TailT = UnionImpl<Index + 1, TailTs...>;
  HeadT Head;
  TailT Tail;
  HeadT &getMember(in_place_index_t<Index>) { return Head; }
  const HeadT &getMember(in_place_index_t<Index>) const { return Head; }
  template <size_t I> decltype(auto) getMember(in_place_index_t<I>) {
    return Tail.getMember(in_place_index<I>);
  }
  template <size_t I> decltype(auto) getMember(in_place_index_t<I>) const {
    return Tail.getMember(in_place_index<I>);
  }
  template <typename... ArgTs>
  UnionImpl(in_place_index_t<Index>, ArgTs &&...Args) {
    new (&Head) HeadT(std::forward<ArgTs>(Args)...);
    Head.IntrusiveVariantTagMember.Index = Index;
  }
  template <size_t I, typename... ArgTs>
  UnionImpl(in_place_index_t<I>, ArgTs &&...Args) {
    new (&Tail) TailT(in_place_index_t<I>{}, std::forward<ArgTs>(Args)...);
  }
  UnionImpl(const UnionImpl &) = default;
  UnionImpl(UnionImpl &&) = default;
  UnionImpl &operator=(const UnionImpl &) = default;
  UnionImpl &operator=(UnionImpl &&) = default;
  // This is safe, assuming the member types are all trivially destructible.
  ~UnionImpl() = default;
};
// The base case for the above, i.e. when the tail pack is empty. This is the
// "(cons head nil)" of the linked list.
template <size_t Index, typename HeadT> union UnionImpl<Index, HeadT> {
  HeadT Head;
  HeadT &getMember(in_place_index_t<Index>) { return Head; }
  const HeadT &getMember(in_place_index_t<Index>) const { return Head; }
  template <typename... ArgTs>
  UnionImpl(in_place_index_t<Index>, ArgTs &&...Args) {
    new (&Head) HeadT(std::forward<ArgTs>(Args)...);
    Head.IntrusiveVariantTagMember.Index = Index;
  }
  UnionImpl(const UnionImpl &) = default;
  UnionImpl(UnionImpl &&) = default;
  UnionImpl &operator=(const UnionImpl &) = default;
  UnionImpl &operator=(UnionImpl &&) = default;
  // This is safe, assuming the member types are all trivially destructible.
  ~UnionImpl() = default;
};
} // end namespace detail

template <typename... Ts> struct VariantTraits<IntrusiveVariant<Ts...>> {
  static constexpr size_t size() { return sizeof...(Ts); }
  static constexpr size_t index(const IntrusiveVariant<Ts...> &Variant) {
    return Variant.index();
  }
  template <size_t Index, typename VariantT = IntrusiveVariant<Ts...>>
  static constexpr decltype(auto) get(VariantT &&Variant) {
    return std::forward<VariantT>(Variant)
        .template get<TypeAtIndex<Index, Ts...>>();
  }
};

/// A class template modeled in the spirit of std::variant, but leveraging the
/// "common initial sequence" rule for union members to store the runtime tag
/// at the beginning of each variant alternative itself, allowing for it to be
/// packed more efficiently into bits that would otherwise be used for padding.
template <typename... Ts> class IntrusiveVariant {
public:
  /// The static minimum number of alternative types supported for an
  /// instantiation of IntrusiveVariant.
  static constexpr size_t MinNumberOfAlternatives = 1;

private:
  static_assert(llvm::conjunction<std::is_standard_layout<Ts>...>::value,
                "IntrusiveVariant alternatives must be standard-layout.");
  static_assert(
      llvm::conjunction<std::is_trivially_destructible<Ts>...>::value,
      "IntrusiveVariant alternatives must be trivially-destructible.");
  template <typename... Us> static constexpr bool tagIsFirstMember() {
    constexpr bool IsFirstMember[] = {
        !offsetof(Us, IntrusiveVariantTagMember)...};
    for (size_t I = 0; I < sizeof...(Us); ++I)
      if (!IsFirstMember[I])
        return false;
    return true;
  }
  /*
  static_assert(
      tagIsFirstMember<Ts...>() &&
          llvm::conjunction<
              std::is_same<IntrusiveVariantTag Ts::*,
                           decltype(&Ts::IntrusiveVariantTagMember)>...>::value,
      "IntrusiveVariant alternatives' class definition must begin with "
      "DECLARE_INTRUSIVE_ALTERNATIVE");
      */
  static_assert(
      TypesAreDistinct<Ts...>::value,
      "Repeated alternative types in IntrusiveVariant are not allowed.");

  // Alias for the UnionImpl of this IntrusiveVariant.
  using UnionT = detail::UnionImpl<0, Ts...>;
  // Helper to get the in_place_index_t for T in Ts...
  template <typename T>
  using InPlaceIndexT = in_place_index_t<FirstIndexOfType<T, Ts...>::value>;
  // Helper to check if a type is in the set Ts...
  template <typename T> using IsAlternativeType = llvm::is_one_of<T, Ts...>;

  // The only data member of IntrusiveVariant, meaning the variant is the same
  // size and has the same alignment requirements as the union of all of its
  // alternative types.
  union {
    detail::CommonInitialSequenceT CommonInitialSequence;
    UnionT Union;
  };

  // Convenience methods to get the union member for an alternative type T.
  template <typename T> T &getAlt() {
    return Union.getMember(InPlaceIndexT<T>{});
  }
  template <typename T> const T &getAlt() const {
    return Union.getMember(InPlaceIndexT<T>{});
  }

public:
  /// A default constructed IntrusiveVariant holds a default constructed value
  /// of its first alternative. Only enabled if the first alternative has a
  /// default constructor.
  template <int B = std::is_default_constructible<TypeAtIndex<0, Ts...>>::value,
            typename std::enable_if_t<B, int> = 0>
  constexpr IntrusiveVariant() : Union(in_place_index_t<0>{}) {}
  /// The forwarding constructor requires a disambiguation tag
  /// in_place_type_t<T>, and creates an IntrusiveVariant holding the
  /// alternative T constructed with the constructor arguments Args...
  template <typename T, std::enable_if_t<IsAlternativeType<T>::value, int> = 0,
            typename... ArgTs>
  explicit constexpr IntrusiveVariant(in_place_type_t<T>, ArgTs &&...Args)
      : Union(InPlaceIndexT<T>{}, std::forward<ArgTs>(Args)...) {}
  /// Converting constructor from alternative types.
  template <typename T, std::enable_if_t<IsAlternativeType<T>::value, int> = 0>
  constexpr IntrusiveVariant(T &&Alt)
      : Union(InPlaceIndexT<T>{}, std::forward<T>(Alt)) {}
  IntrusiveVariant(const IntrusiveVariant &) = default;
  IntrusiveVariant(IntrusiveVariant &&) = default;
  ~IntrusiveVariant() = default;
  IntrusiveVariant &operator=(const IntrusiveVariant &) = default;
  IntrusiveVariant &operator=(IntrusiveVariant &&) = default;
  /// Replaces the held value with a new value of alternative type T in-place,
  /// constructing the new value with constructor arguments Args...
  ///
  /// Returns the newly constructed alternative type value.
  template <typename T, typename... ArgTs> T &emplace(ArgTs &&...Args) {
    new (&Union) UnionT(InPlaceIndexT<T>{}, std::forward<ArgTs>(Args)...);
    return Union.getMember(InPlaceIndexT<T>{});
  }
  /// Returns the index of the alternative type held by this variant.
  size_t index() const { return CommonInitialSequence.Tag.Index; }
  /// Check if this variant holds a value of the given alternative type T.
  template <class T> constexpr bool holdsAlternative() const {
    return index() == FirstIndexOfType<T, Ts...>();
  }
  /// Reads the value of alternative type T.
  ///
  /// Behavior undefined if this does not hold a value of alternative type T.
  template <class T> constexpr T &get() {
    assert(holdsAlternative<T>());
    return getAlt<T>();
  }
  /// Reads the value of alternative type T.
  ///
  /// Behavior undefined if this does not hold a value of alternative type T.
  template <class T> constexpr const T &get() const {
    assert(holdsAlternative<T>());
    return getAlt<T>();
  }
  /// Obtains a pointer to the value of alternative type T if this holds a
  /// value of alternative type T. Otherwise, returns nullptr.
  template <class T> constexpr T *getIf() {
    if (holdsAlternative<T>())
      return &getAlt<T>();
    return nullptr;
  }
  /// Obtains a pointer to the value of alternative type T if this holds a
  /// value of alternative type T. Otherwise, returns nullptr.
  template <class T> constexpr const T *getIf() const {
    if (holdsAlternative<T>())
      return &getAlt<T>();
    return nullptr;
  }

  /// Equality operator.
  ///
  /// The alternative types held by LHS and RHS are T and U, respectively; then:
  ///
  /// If T != U, returns false.
  /// Otherwise, returns LHS.get<T>() == RHS.get<U>().
  friend constexpr bool operator==(const IntrusiveVariant<Ts...> &LHS,
                                   const IntrusiveVariant<Ts...> &RHS) {
    if (LHS.index() != RHS.index())
      return false;
    return visitSameAlternative(std::equal_to<>{}, LHS, RHS);
  }

  /// Inequality operator.
  ///
  /// The alternative types held by LHS and RHS are T and U, respectively; then:
  ///
  /// If T != U, returns true.
  /// Otherwise, returns LHS.get<T>() != RHS.get<U>().
  friend constexpr bool operator!=(const IntrusiveVariant<Ts...> &LHS,
                                   const IntrusiveVariant<Ts...> &RHS) {
    if (LHS.index() != RHS.index())
      return true;
    return visitSameAlternative(std::not_equal_to<>{}, LHS, RHS);
  }

  /// Less-than operator.
  ///
  /// The alternative types held by LHS and RHS are T and U, respectively; then:
  ///
  /// If T precedes U in Ts..., returns true.
  /// If U precedes T in Ts..., returns false.
  /// Otherwise, returns LHS.get<T>() < RHS.get<U>().
  friend constexpr bool operator<(const IntrusiveVariant<Ts...> &LHS,
                                  const IntrusiveVariant<Ts...> &RHS) {
    if (LHS.index() < RHS.index())
      return true;
    if (LHS.index() > RHS.index())
      return false;
    return visitSameAlternative(std::less<>{}, LHS, RHS);
  }

  /// Greater-than operator.
  ///
  /// The alternative types held by LHS and RHS are T and U, respectively; then:
  ///
  /// If T precedes U in Ts..., returns false.
  /// If U precedes T in Ts..., returns true.
  /// Otherwise, returns LHS.get<T>() > RHS.get<U>().
  friend constexpr bool operator>(const IntrusiveVariant<Ts...> &LHS,
                                  const IntrusiveVariant<Ts...> &RHS) {
    if (LHS.index() < RHS.index())
      return false;
    if (LHS.index() > RHS.index())
      return true;
    return visitSameAlternative(std::greater<>{}, LHS, RHS);
  }

  /// Less-equal operator.
  ///
  /// The alternative types held by LHS and RHS are T and U, respectively; then:
  ///
  /// If T precedes U in Ts..., returns true.
  /// If U precedes T in Ts..., returns false.
  /// Otherwise, returns LHS.get<T>() <= RHS.get<U>().
  friend constexpr bool operator<=(const IntrusiveVariant<Ts...> &LHS,
                                   const IntrusiveVariant<Ts...> &RHS) {
    if (LHS.index() < RHS.index())
      return true;
    if (LHS.index() > RHS.index())
      return false;
    return visitSameAlternative(std::less_equal<>{}, LHS, RHS);
  }

  /// Greater-equal operator.
  ///
  /// The alternative types held by LHS and RHS are T and U, respectively; then:
  ///
  /// If T precedes U in Ts..., returns false.
  /// If U precedes T in Ts..., returns true.
  /// Otherwise, returns LHS.get<T>() >= RHS.get<U>().
  friend constexpr bool operator>=(const IntrusiveVariant<Ts...> &LHS,
                                   const IntrusiveVariant<Ts...> &RHS) {
    if (LHS.index() < RHS.index())
      return false;
    if (LHS.index() > RHS.index())
      return true;
    return visitSameAlternative(std::greater_equal<>{}, LHS, RHS);
  }

  /// Enabled if all alternative types overload hash_value.
  friend hash_code hash_value(const IntrusiveVariant &IV) {
    return visit(
        [&](auto &&Alt) { return hash_combine(IV.index(), hash_value(Alt)); },
        IV);
  }
};

} // end namespace llvm

#endif // LLVM_ADT_INTRUSIVEVARIANT_H
