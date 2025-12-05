//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implement common constexpr utilities including sorting and integer
/// sequence manipulation.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_CONSTEXPRUTILS_H
#define LLVM_ADT_CONSTEXPRUTILS_H

#include "llvm/Support/MathExtras.h"

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace llvm {

/// This is a tag class returned by meta-programming to indicate an error.
///
/// Users may derive from this class to provide more context on the failure.
///
/// Using a tag class allows the infrastructure to `static_assert` at higher,
/// more relevant scopes and generates much more friendly template errors.
struct CETypeError {};

namespace constexpr_detail {

template <std::size_t I, typename... Tn> struct PackElement {};

template <std::size_t I, typename T0, typename... Tn>
struct PackElement<I, T0, Tn...> {
  static_assert(I <= sizeof...(Tn), "Pack index out of bounds.");

  using type = typename PackElement<I - 1, Tn...>::type;
};

template <typename T0, typename... Tn> struct PackElement<0, T0, Tn...> {
  using type = T0;
};

/// An extraction cannot occur from an empty type pack.
struct PackElementEmptyPack : CETypeError {};

template <std::size_t I> struct PackElement<I> {
  using type = PackElementEmptyPack;
};

} // namespace constexpr_detail

template <std::size_t Idx, typename... Tn>
using PackElementT = typename constexpr_detail::PackElement<Idx, Tn...>::type;

/// Swap values of two specified references at compile-time.
template <typename T>
constexpr void ce_swap(T &L, T &R) // NOLINT (readability-identifier-naming)
{
  T Temp = L;
  L = R;
  R = Temp;
}

/// Compute the minimum value from a list of compile-time constants.
///
/// \tparam RetT  Type to return and `static_cast` each value to for comparison.
///
/// \param Val0   First value in the list to compute the minimum of.
/// \param ValN   Remaining values in the list to compute the minimum of.
template <typename RetT, typename T0, typename... Tn>
constexpr RetT                //
ce_min(T0 Val0, Tn... ValN) { // NOLINT(readability-identifier-naming)
  RetT Min = static_cast<RetT>(Val0);
  ((Min = static_cast<RetT>(ValN) < Min ? static_cast<RetT>(ValN) : Min), ...);
  return Min;
}

/// Compute the maximum value from a list of compile-time constants.
///
/// \tparam RetT  Type to return and `static_cast` each value to for comparison.
///
/// \param Val0  First value in the list to compute the maximum of.
/// \param ValN  Remaining values in the list to compute the maximum of.
template <typename RetT, typename T0, typename... Tn>
constexpr RetT                //
ce_max(T0 Val0, Tn... ValN) { // NOLINT(readability-identifier-naming)
  RetT Max = static_cast<RetT>(Val0);
  ((Max = static_cast<RetT>(ValN) > Max ? static_cast<RetT>(ValN) : Max), ...);
  return Max;
}

namespace constexpr_detail {

template <typename T, std::size_t N, std::size_t... I>
constexpr auto toArray(T const (&Arr)[N], std::index_sequence<I...>) {
  return std::array<std::remove_cv_t<T>, N>{{Arr[I]...}};
}

} // namespace constexpr_detail

/// Convert a C array to an equivalent `std::array`.
template <typename T, std::size_t N>
constexpr auto
to_array(T const (&Arr)[N]) // NOLINT (readability-identifier-naming)
{
  return constexpr_detail::toArray(Arr, std::make_index_sequence<N>());
}

/// Convert a `std::integer_sequence` to a `std::array` with the same contents.
template <typename T, T... Vals>
constexpr auto to_array( // NOLINT (readability-identifier-naming)
    std::integer_sequence<T, Vals...>) {
  return std::array<T, sizeof...(Vals)>{{Vals...}};
}

namespace constexpr_detail {

template <typename IterT, typename LessT>
constexpr void bubbleSort(IterT B, IterT E, LessT const &Less) {
  for (auto Last = E, ELast = std::next(B); Last != ELast; --Last) {
    bool Swapped = false;

    for (auto I = ELast; I != Last; ++I) {
      if (Less(*I, *std::prev(I))) {
        ce_swap(*I, *std::prev(I));
        Swapped = true;
      }
    }

    // Remaining items are in sorted order if no swaps occurred during the
    // inner loop.
    if (!Swapped)
      return;
  }
}

template <typename T, std::size_t N, typename LessT>
constexpr T &quickSortGetPivot(std::array<T, N> &Arr, std::size_t Low,
                               std::size_t High, LessT const &Less) {
  std::size_t Middle = Low + (((High - Low) + 1) >> 1);
  // Use "median of three" (repositioned into the 'High' position) as pivot...
  if (Less(Arr[High], Arr[Low]))
    ce_swap(Arr[High], Arr[Low]);

  if (Middle != High) {
    if (Less(Arr[Middle], Arr[Low]))
      ce_swap(Arr[Middle], Arr[Low]);

    if (Less(Arr[Middle], Arr[High]))
      ce_swap(Arr[Middle], Arr[High]);
  }
  return Arr[High];
}

template <typename T, std::size_t N, typename LessT>
constexpr std::size_t quickSortPartition(std::array<T, N> &Arr, std::size_t Low,
                                         std::size_t High, LessT const &Less) {
  auto &Pivot = quickSortGetPivot(Arr, Low, High, Less);

  std::size_t I = Low;
  for (std::size_t J = Low; J < High; ++J)
    if (Less(Arr[J], Pivot))
      ce_swap(Arr[I++], Arr[J]);

  ce_swap(Arr[I], Arr[High]);
  return I;
}

struct QuickSortIndexPair {
  std::size_t Low = 0, High = 0;
  constexpr void set(std::size_t L, std::size_t H) { Low = L, High = H; }
};

template <std::size_t N> class QuickSortStack {
  std::size_t Size = 0;
  std::array<QuickSortIndexPair, N> Arr;

public:
  constexpr bool empty() const { return Size == 0; }
  constexpr void push(std::size_t L, std::size_t H) { Arr[Size++].set(L, H); }
  constexpr QuickSortIndexPair pop() { return Arr[--Size]; }
  constexpr std::size_t availableSlots() const { return N - Size; }
};

// Use bubble sort on partitions smaller than this size.  Bubble will be
// more efficient on small partitions.
inline constexpr std::size_t MinQuickSortPartitionSize = 10u;

template <typename T, std::size_t N, typename LessT>
constexpr void quickSort(std::array<T, N> &Arr, LessT const &Less) {
  if constexpr (N <= 1)
    return;

  // The maximum stack depth is O(log N) on average.  O(N) is possible in
  // corner-cases, so the partition loop will fall back to bubble sorting
  // partitions if the stack is exhausted.
  constexpr std::size_t StackSize = ConstantLog2<NextPowerOf2(N)>() + 1;
  constexpr_detail::QuickSortStack<StackSize> Stack;

  Stack.push(0, N - 1);

  while (!Stack.empty()) {
    auto const [Low, High] = Stack.pop();

    if ((High - Low) + 1u < MinQuickSortPartitionSize ||
        Stack.availableSlots() < 2u) {
      bubbleSort(Arr.begin() + Low, Arr.begin() + High + 1, Less);
      continue;
    }

    std::size_t const PivotIndex =
        constexpr_detail::quickSortPartition(Arr, Low, High, Less);

    std::size_t PivotIndexLo = PivotIndex - 1;
    std::size_t PivotIndexHi = PivotIndex + 1;

    // Pivot is in sorted position; so are any adjacent elements which are
    // equivalent.  Scan past elements which are equal to the pivot before
    // sub-dividing the current range.
    while (PivotIndexLo > Low && PivotIndexLo < High &&
           !Less(Arr[PivotIndexLo], Arr[PivotIndex]))
      --PivotIndexLo;
    while (PivotIndexHi < High && !Less(Arr[PivotIndex], Arr[PivotIndexHi]))
      ++PivotIndexHi;

    // Push right sub-array boundaries to the Stack if it exists
    if (High > PivotIndexHi)
      Stack.push(PivotIndexHi, High);

    // Push left sub-array boundaries to the Stack if it exists
    if (Low < PivotIndexLo && PivotIndexLo < N)
      Stack.push(Low, PivotIndexLo);
  }
}

} // namespace constexpr_detail

/// Iterative sort implementation.
///
/// \param[in,out] Arr  Array of elements to sort in-place.
/// \param Less  Less-than comparison functor which compares two `T`s.
template <typename T, std::size_t N, typename LessT>
constexpr void ce_sort_inplace( // NOLINT (readability-identifier-naming)
    std::array<T, N> &Arr, LessT const &Less) {
  if constexpr (N >= constexpr_detail::MinQuickSortPartitionSize)
    constexpr_detail::quickSort(Arr, Less);
  else if (N > 1)
    // Avoid instantiating the quickSort stack if we won't even parition
    // elements once.
    constexpr_detail::bubbleSort(Arr.begin(), Arr.end(), Less);
}

/// Iterative sort implementation.
///
/// \param[in,out] Arr  Array of elements to sort in-place.
template <typename T, std::size_t N>
constexpr void
ce_sort_inplace(std::array<T, N> &Arr) // NOLINT (readability-identifier-naming)
{
  constexpr auto Less = [](T const &L, T const &R) constexpr { return L < R; };
  ce_sort_inplace(Arr, Less);
}

/// Iterative sort implementation.
///
/// The sort implementation is chosen at compile-time based on the size of the
/// specified array.
///
/// \param ArrIn Input sequence to sort.
/// \param Less  Less-than comparison functor which compares two `T`s.
///
/// \return A `std::array` of elements in ascending order according to the
/// specified \p Less comparison.
template <typename T, std::size_t N, typename LessT>
constexpr std::array<std::remove_cv_t<T>, N>
ce_sort( // NOLINT (readability-identifier-naming)
    T const (&ArrIn)[N], LessT const &Less) {
  auto Arr = to_array(ArrIn);
  ce_sort_inplace(Arr, Less);
  return Arr;
}

/// Iterative sort implementation.
///
/// The sort implementation is chosen at compile-time based on the size of the
/// specified array.
///
/// \param ArrIn Input sequence to sort.
///
/// \return A `std::array` of elements in ascending order according based on the
/// `operator<()` associated with `T`.
template <typename T, std::size_t N>
constexpr std::array<std::remove_cv_t<T>, N>
ce_sort(T const (&ArrIn)[N]) // NOLINT (readability-identifier-naming)
{
  auto Arr = to_array(ArrIn);
  ce_sort_inplace(Arr);
  return Arr;
}

namespace constexpr_detail {

template <typename T, T... Is> class SortLiteralsImpl {
  static constexpr auto makeSortedArray() {
    std::array<T, sizeof...(Is)> Arr{{Is...}};
    ce_sort_inplace(Arr);
    return Arr;
  }

  // Helper to reconstruct integer_sequence from the sorted array
  template <std::size_t... Idxs>
  static constexpr auto makeSortedSequence(std::index_sequence<Idxs...>) {
    constexpr auto SortedArray = makeSortedArray();
    return std::integer_sequence<T, SortedArray[Idxs]...>{};
  }

public:
  using type =
      decltype(makeSortedSequence(std::make_index_sequence<sizeof...(Is)>()));
};

} // namespace constexpr_detail

/// Create a sorted `std::integer_sequence` from a list of literals.
///
/// \tparam T  Integral type of output `std::integer_sequence` as well as all
/// the specified literals.
/// \tparam Is  List of literals to sort.
template <typename T, T... Is>
using SortLiterals =
    typename constexpr_detail::SortLiteralsImpl<T, Is...>::type;

namespace constexpr_detail {

template <typename SeqT> class SortSequenceImpl {};

template <typename T, T... Is>
class SortSequenceImpl<std::integer_sequence<T, Is...>> {
public:
  using type = typename SortLiteralsImpl<T, Is...>::type;
};

} // namespace constexpr_detail

/// Create a sorted `std::integer_sequence` from a specified
/// `std::integer_sequence`.
///
/// \tparam SeqT  `std::integer_sequence` which specifies the literals to be
/// sorted.
template <typename SeqT>
using SortSequence = typename constexpr_detail::SortSequenceImpl<SeqT>::type;

namespace constexpr_detail {

template <std::size_t First, typename T, std::size_t N, std::size_t... Idx>
constexpr auto ce_slice_impl( // NOLINT (readability-identifier-naming)
    std::array<T, N> const &Arr, std::index_sequence<Idx...>) {
  return std::array<T, sizeof...(Idx)>{{Arr[First + Idx]...}};
}

} // namespace constexpr_detail

/// Create a `std::array` representing a slice of an input `std::array`.
///
/// \tparam First Index of the first element of the output slice.
/// \tparam Len   Number of elements int the output slice.
///
/// \param Arr    Array to slice from.
///
/// \return A new `std::array` of length `Len` representing the specified slice
/// of \p Arr.
template <std::size_t First, std::size_t Len, typename T, std::size_t N>
constexpr auto
ce_slice(std::array<T, N> const &Arr) // NOLINT (readability-identifier-naming)
{
  static_assert(First + Len <= N, "End of slice is out of bounds.");
  return constexpr_detail::ce_slice_impl<First>(
      Arr, std::make_index_sequence<Len>());
}

namespace constexpr_detail {

template <std::size_t First, std::size_t Len, typename T, T... Is>
class SliceLiteralsImpl {
  static_assert(First + Len <= sizeof...(Is), "End of slice is out of bounds.");

  template <std::size_t... Idxs>
  static constexpr auto makeSlicedSequence(std::index_sequence<Idxs...>) {
    constexpr std::array<T, sizeof...(Is)> InArr{{Is...}};
    return std::integer_sequence<T, InArr[First + Idxs]...>{};
  }

public:
  using type = decltype(makeSlicedSequence(std::make_index_sequence<Len>()));
};

template <std::size_t First, typename T, T... Is>
class SliceLiteralsImpl<First, 0, T, Is...> {
  static_assert(First <= sizeof...(Is));

public:
  using type = std::integer_sequence<T>;
};

} // namespace constexpr_detail

/// Create a `std::index_sequence` representing a slice of list of literals.
///
/// \tparam First Index of the first element of the output slice.
/// \tparam Len   Number of elements int the output slice.
/// \tparam T     Type of the literals being sliced.
/// \tparam Is    List of literals to slice.
template <std::size_t First, std::size_t Len, typename T, T... Is>
using SliceLiterals =
    typename constexpr_detail::SliceLiteralsImpl<First, Len, T, Is...>::type;

namespace constexpr_detail {

template <std::size_t First, std::size_t Len, typename SeqT>
class SliceSequenceImpl {};

template <std::size_t First, std::size_t Len, typename T, T... Is>
class SliceSequenceImpl<First, Len, std::integer_sequence<T, Is...>> {
public:
  using type = SliceLiterals<First, Len, T, Is...>;
};

} // namespace constexpr_detail

/// Generate a `std::integer_sequence` representing a sub-range of a specified
/// `std::integer_sequence`.
///
/// \tparam First Index of the first element of the output slice.
/// \tparam Len   Number of elements int the output slice.
/// \tparam SeqT  `std::integer_sequence` to slice from.
template <std::size_t First, std::size_t Len, typename SeqT>
using SliceSequence =
    typename constexpr_detail::SliceSequenceImpl<First, Len, SeqT>::type;

namespace constexpr_detail {

template <auto Val, typename SeqT> class PushBackSequenceImpl {};

template <auto Val, typename T, T... Is>
class PushBackSequenceImpl<Val, std::integer_sequence<T, Is...>> {
public:
  using type = std::integer_sequence<T, Is..., T(Val)>;
};

} // namespace constexpr_detail

/// Add an integer value to the end of an `std::integer_sequence`.
///
/// \tparam SeqT `std::integer_sequence` to append to.
/// \tparam Val  Integral value to append to \p SeqT.
template <typename SeqT, auto Val>
using PushBackSequence =
    typename constexpr_detail::PushBackSequenceImpl<Val, SeqT>::type;

} // namespace llvm

#endif // LLVM_ADT_CONSTEXPRUTILS_H
