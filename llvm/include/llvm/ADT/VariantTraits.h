//===- VariantTraits.h - Common interfaces for variant-like types --C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains common interfaces for variant-like types.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"

#ifndef LLVM_ADT_VARIANTTRAITS_H
#define LLVM_ADT_VARIANTTRAITS_H

namespace llvm {

/// Trait type which can be specialized over std::variant-like types to provide
/// the minimum interface needed to share the implementation of llvm::visit and
/// llvm::visitSameAlternative.
template <typename VariantT> struct VariantTraits {
  // // Returns the number of alternative types of VariantT.
  // static constexpr size_t size();
  //
  // // Returns the index of the current alternative type of Variant.
  // static constexpr size_t index(const VariantT &Variant);
  //
  // // Gets the alternative type at Index.
  // template <size_t Index, typename VariantT = VariantT>
  // static constexpr decltype(auto) get(VariantT &&Variant);
};

namespace variant_traits_detail {

template <typename T> using Traits = struct VariantTraits<remove_cvref_t<T>>;

template <typename T> struct HasTraits {
  using Absent = char;
  using Present = long;
  template <typename U> static Absent size(...);
  template <typename U> static Present size(SameType<size_t (*)(), &U::size> *);
  template <typename U> static Absent index(...);
  template <typename U>
  static Present
  index(SameType<size_t (*)(const remove_cvref_t<T> &), &U::index> *);
  template <typename U> static Absent get(...);
  template <typename U, typename R>
  static Present get(SameType<R (*)(remove_cvref_t<T> &&), &U::get> *);

  static bool const value = // NOLINT(readability-identifier-naming)
      sizeof(size<Traits<T>>(nullptr)) == sizeof(Present) &&
      sizeof(index<Traits<T>>(nullptr)) == sizeof(Present) &&
      sizeof(get<Traits<T>>(nullptr) == sizeof(Present));
};

template <size_t Index, typename VisitorT, typename... VariantTs>
static constexpr decltype(auto)
thunkForSameAlternative(VisitorT &&Visitor, VariantTs &&...Variants) {
  return std::forward<VisitorT>(Visitor)(Traits<VariantTs>::template get<Index>(
      std::forward<VariantTs>(Variants))...);
}

template <size_t Index, typename VisitorT, typename... VariantTs>
static constexpr auto makeThunkForSameAlternative() {
  return thunkForSameAlternative<Index, VisitorT, VariantTs...>;
}

template <typename VisitorT, typename HeadVariantT, typename... TailVariantTs,
          size_t... Indexes>
static constexpr auto
visitSameAlternativeImpl(size_t Index, std::index_sequence<Indexes...>,
                         VisitorT &&Visitor, HeadVariantT &&HeadVariant,
                         TailVariantTs &&...TailVariants) {
  constexpr auto Thunks =
      make_array(makeThunkForSameAlternative<Indexes, VisitorT, HeadVariantT,
                                             TailVariantTs...>()...);
  return Thunks[Index](std::forward<VisitorT>(Visitor),
                       std::forward<HeadVariantT>(HeadVariant),
                       std::forward<TailVariantTs>(TailVariants)...);
}

template <size_t... Indexes> struct Thunk {
  template <typename VisitorT, typename... VariantTs>
  inline static constexpr decltype(auto) thunk(VisitorT &&Visitor,
                                               VariantTs &&...Variants) {
    return std::forward<VisitorT>(Visitor)(
        Traits<VariantTs>::template get<Indexes>(
            std::forward<VariantTs>(Variants))...);
  }
};

template <typename VisitorT, typename... VariantTs, size_t... Indexes>
static constexpr auto makeThunkForSequence(std::index_sequence<Indexes...>) {
  return Thunk<Indexes...>::template thunk<VisitorT, VariantTs...>;
}

template <typename VisitorT, typename... VariantTs,
          size_t... AccumulatedIndexes>
static constexpr auto
accumulateCartesianProductThunks(std::index_sequence<AccumulatedIndexes...>) {
  return makeThunkForSequence<VisitorT, VariantTs...>(
      std::index_sequence<AccumulatedIndexes...>{});
}

template <typename VisitorT, typename... VariantTs,
          size_t... AccumulatedIndexes, size_t... HeadIndexes,
          typename... TailSequenceTs>
static constexpr auto
accumulateCartesianProductThunks(std::index_sequence<AccumulatedIndexes...>,
                                 std::index_sequence<HeadIndexes...>,
                                 TailSequenceTs... Tail) {
  return make_array(accumulateCartesianProductThunks<VisitorT, VariantTs...>(
      std::index_sequence<AccumulatedIndexes..., HeadIndexes>{}, Tail...)...);
}

template <typename VisitorT, typename... VariantTs>
static constexpr auto makeThunkMatrix() {
  return accumulateCartesianProductThunks<VisitorT, VariantTs...>(
      std::index_sequence<>{},
      std::make_index_sequence<Traits<VariantTs>::size()>{}...);
}

template <typename ThunkT>
static constexpr const ThunkT &indexThunkMatrix(const ThunkT &Thunk) {
  return Thunk;
}

template <typename ThunkMatrixT, typename... TailIndexTs>
static constexpr auto &&indexThunkMatrix(const ThunkMatrixT &ThunkMatrix,
                                         size_t HeadIndex,
                                         TailIndexTs... TailIndexes) {
  return indexThunkMatrix(ThunkMatrix[HeadIndex], TailIndexes...);
}

} // namespace variant_traits_detail

/// Invokes the provided Visitor using overload resolution based on the
/// dynamic alternative type held in each Variant. See std::variant.
///
/// The return type is effectively
/// decltype(Visitor(Variants.get<HeldAlternatives>()...)). This must be a
/// valid expression of the same type and value category for every combination
/// of alternative types of the variant types.
template <
    typename VisitorT, typename... VariantTs,
    typename std::enable_if_t<
        conjunction<variant_traits_detail::HasTraits<VariantTs>...>::value,
        int> = 0>
constexpr decltype(auto) visit(VisitorT &&Visitor, VariantTs &&...Variants) {
  constexpr auto ThunkMatrix =
      variant_traits_detail::makeThunkMatrix<VisitorT, VariantTs...>();
  const auto &Thunk = variant_traits_detail::indexThunkMatrix(
      ThunkMatrix, variant_traits_detail::Traits<VariantTs>::index(
                       std::forward<VariantTs>(Variants))...);
  return Thunk(std::forward<VisitorT>(Visitor),
               std::forward<VariantTs>(Variants)...);
}

/// Invokes the provided Visitor using overload resolution based on the dynamic
/// alternative type held in each Variant, assuming the variants are all of the
/// same type and hold the same dynamic alternative type.
///
/// \warning llvm::visit must be used instead when there is no guarantee that
/// all variants currently hold the same alternative type. However, when such a
/// guarantee can be made llvm::visitSameAlternative may reduce code bloat,
/// especially for debug builds.
///
/// The return type is effectively
/// decltype(Visitor(Variants.get<HeldAlternative>()...)). This must be a valid
/// expression of the same type and value category for every alternative type
/// of the variant type.
template <
    typename VisitorT, typename HeadVariantT, typename... TailVariantTs,
    typename std::enable_if_t<
        conjunction<variant_traits_detail::HasTraits<HeadVariantT>,
                    variant_traits_detail::HasTraits<TailVariantTs>...>::value,
        int> = 0>
static constexpr decltype(auto)
visitSameAlternative(VisitorT &&Visitor, HeadVariantT &&HeadVariant,
                     TailVariantTs &&...TailVariants) {
  static_assert(
      conjunction<std::is_same<remove_cvref_t<HeadVariantT>,
                               remove_cvref_t<TailVariantTs>>...>::value,
      "all variant arguments to visitSameAlternative must "
      "be of the same type");
  using Traits = variant_traits_detail::Traits<HeadVariantT>;
#ifdef EXPENSIVE_CHECKS
  size_t Index = Traits::index(std::forward<HeadVariantT>(HeadVariant));
  for (auto &&V : {std::forward<TailVariantTs>(TailVariants)...})
    assert(Traits::index(V) == Index &&
           "all variant arguments to visitSameAlternative must have "
           "the same index");
#endif
  return variant_traits_detail::visitSameAlternativeImpl(
      Traits::index(std::forward<HeadVariantT>(HeadVariant)),
      std::make_index_sequence<Traits::size()>{},
      std::forward<VisitorT>(Visitor), std::forward<HeadVariantT>(HeadVariant),
      std::forward<TailVariantTs>(TailVariants)...);
}

} // namespace llvm

#endif // LLVM_ADT_VARIANTTRAITS_H
