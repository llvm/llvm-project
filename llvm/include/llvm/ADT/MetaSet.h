//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implements template meta-programming set representations for use in
/// `constexpr` expressions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_METASET_H
#define LLVM_ADT_METASET_H

#include "llvm/ADT/ConstexprUtils.h"
#include "llvm/ADT/bit.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace llvm {

/// Max number of words allowed in a MetaBitset.
///
/// This limit prevents unbounded host compile-time / space in .rodata.
static constexpr size_t MetaBitsetWordLimit = 512;

struct MetaSetTag {};
struct MetaSetDuplicateElemsTag {};
struct MetaBitsetTag : MetaSetTag {};
struct MetaSequenceSetTag : MetaSetTag, MetaSetDuplicateElemsTag {};
struct MetaSparseBitsetTag : MetaSetTag {};

template <typename T>
inline constexpr bool IsMetaSet = std::is_base_of_v<MetaSetTag, T>;

template <typename T>
inline constexpr bool IsMetaBitset = std::is_base_of_v<MetaBitsetTag, T>;

template <typename T>
inline constexpr bool IsMetaSequenceSet =
    std::is_base_of_v<MetaSequenceSetTag, T>;

template <typename T>
inline constexpr bool IsMetaSparseBitset =
    std::is_base_of_v<MetaSparseBitsetTag, T>;

template <typename T>
inline constexpr bool IsMetaSetWithDuplicateElements =
    std::is_base_of_v<MetaSetDuplicateElemsTag, T>;

namespace metaset_detail {

template <typename PosT, PosT Offset> struct TypeInfo {
  static_assert(std::is_integral_v<PosT>,
                "The position type of a `MetaBitset` must be integral.");

  using WordT = uint64_t;
  using UPosT = std::make_unsigned_t<PosT>;
  using WordIndexT = UPosT;

  static constexpr UPosT WordNBits = 64;
  static constexpr UPosT IdxShift = 6; // log2(64)

  class NormalizedT {
    UPosT NormalizedPos;

    constexpr NormalizedT(PosT Pos) : NormalizedPos(UPosT(Pos - Offset)) {}

    // Allow TypeInfo access to the constructor.
    friend struct TypeInfo;

  public:
    constexpr UPosT wordIndex() const { return NormalizedPos >> IdxShift; }

    constexpr WordT bitMask() const {
      constexpr UPosT BitIdxMask = (UPosT(1) << IdxShift) - UPosT(1);
      return WordT(1) << (NormalizedPos & BitIdxMask);
    }
  };

  static constexpr NormalizedT normalizePos(PosT Pos) { return Pos; }
};

} // namespace metaset_detail

/// Represent a set of integral values using a dense bitset representation.
///
/// The `MetaBitset` has no storage on the class itself, but it refers to a
/// `static constexpr` array of words that represent the elements in the set.
///
/// This class must be built with helper type aliases.  See `MakeMetaBitset`
/// and `MakeMetaBitsetFromSequence`.  If converting a sorted
/// `std::integer_sequence` to a `MetaBitset`, use
/// `MakeMetaBitsetFromSortedSequence` to save some build time.
///
/// \tparam PosT  Integral or enum type used to represent the position / offset
/// in the bitset.
/// \tparam Offset  Adjustment applied to all positions in the bitset, thus
/// enabling efficient representation of clusters of large numbers.
/// \tparam Words  The unsigned words whose bits represent the set.
template <typename PosT, PosT Offset, uint64_t... Words>
class MetaBitset : metaset_detail::TypeInfo<PosT, Offset>,
                   public MetaBitsetTag {
  using WordT = typename MetaBitset::TypeInfo::WordT;
  using UPosT = typename MetaBitset::TypeInfo::UPosT;

  static constexpr std::array<WordT, sizeof...(Words)> WordArr{{Words...}};

  static_assert(WordArr.size() == 0 || WordArr[0] != 0ULL,
                "A MetaBitset is malformed if its low-order word is zero.");

public:
  using value_type = PosT;

public:
  constexpr MetaBitset() {}

  static constexpr std::size_t count() { return (0u + ... + popcount(Words)); }

  static constexpr bool empty() { return sizeof...(Words) == 0; }

  /// Determine whether \p Pos is equivalent to position with a `1` in the
  /// bitset.
  ///
  /// \return `true` if \p Pos is in the set; `false` otherwise.
  ///
  /// \note Complexity = `O(1)`.
  template <typename T> static constexpr bool contains(T Pos) {
    auto NPos = MetaBitset::normalizePos(PosT(Pos));

    UPosT Idx = NPos.wordIndex();
    if (Idx >= sizeof...(Words))
      return false;

    return WordArr[Idx] & NPos.bitMask();
  }

  static constexpr auto to_array() // NOLINT (readability-identifier-naming)
  {
    return toArrayImpl<count()>();
  }

  static constexpr auto
  to_sorted_array() // NOLINT (readability-identifier-naming)
  {
    return to_array();
  }

private:
  template <std::size_t N> static constexpr std::array<PosT, N> toArrayImpl() {
    std::array<PosT, N> Arr{{}};

    std::size_t SeqIdx = 0;
    for (std::size_t I = 0, E = WordArr.size(); I != E; ++I) {
      uint64_t Word = WordArr[I];
      std::size_t WordBaseIdx = Offset + (I << 6);

      std::size_t B = countr_zero_constexpr(Word);
      while (B < MetaBitset::TypeInfo::WordNBits) {
        Arr[SeqIdx++] = WordBaseIdx + B;
        Word &= ~(1ULL << B);
        B = countr_zero_constexpr(Word);
      }
    }
    return Arr;
  }

  template <std::size_t... I>
  static constexpr auto toSequenceImpl(std::index_sequence<I...> IdxSeq) {
    constexpr auto SeqArr = toArrayImpl<IdxSeq.size()>();
    return std::integer_sequence<PosT, SeqArr[I]...>();
  }

public:
  using sequence_type =
      decltype(toSequenceImpl(std::make_index_sequence<count()>()));
  using sorted_sequence_type = sequence_type;
};

/// Compute the number of words a `MetaBitset` would need to represent a set
/// with the specified \p MinVal and \p MaxVal.
///
/// \tparam MinVal  The minimum value in the set to be represented as a
/// `MetaBitset`.
/// \tparam MaxVal  The maximum value in the set to be represented as a
/// `MetaBitset`.
template <typename PosT, PosT MinVal, PosT MaxVal>
inline constexpr size_t MetaBitsetNumWordsDetailed =
    (size_t(MaxVal - MinVal) + 64U) / 64U;

namespace metaset_detail {

template <typename PosT, PosT MinVal, PosT MaxVal, PosT... Values>
struct MakeMetaBitsetImpl {
  using TypeInfo = metaset_detail::TypeInfo<PosT, MinVal>;

  static constexpr size_t NumWords =
      MetaBitsetNumWordsDetailed<PosT, MinVal, MaxVal>;

  static_assert(NumWords < MetaBitsetWordLimit,
                "MetaBitset exceeds word limit (llvm::MetaBitsetWordLimit).");

  static constexpr auto makeWordsArray() {
    constexpr std::size_t NumValues = sizeof...(Values);
    constexpr std::array<PosT, NumValues> BitPositions{
        {static_cast<PosT>(Values)...}};
    std::array<uint64_t, NumWords> Words{};
    for (PosT BitPos : BitPositions) {
      auto NPos = TypeInfo::normalizePos(BitPos);
      Words[NPos.wordIndex()] |= NPos.bitMask();
    }
    return Words;
  }

  template <size_t... Is>
  static constexpr auto makeBitset(std::index_sequence<Is...>) {
    constexpr auto WordsArr = makeWordsArray();
    return MetaBitset<PosT, MinVal, WordsArr[Is]...>();
  }

public:
  using type = decltype(makeBitset(std::make_index_sequence<NumWords>()));
};

template <typename T> struct MetaBitsetNumWordsImpl;

template <typename PosT, PosT Offset, uint64_t... Words>
struct MetaBitsetNumWordsImpl<MetaBitset<PosT, Offset, Words...>>
    : std::integral_constant<size_t, sizeof...(Words)> {};

} // namespace metaset_detail

/// Get the number of words used to represent a `MetaBitset`.
template <typename MetaBitsetT>
inline constexpr size_t MetaBitsetNumWords =
    metaset_detail::MetaBitsetNumWordsImpl<MetaBitsetT>::value;

/// Create a `MetaBitset` containing the specified values with the specified
/// bounds.
///
/// \tparam PosT  Bit-position type to apply to the `MetaBitset`.
/// \tparam MinVal  The smallest value represented in the `MetaBitset`.
/// \tparam MaxVal  The largest value represnted in the `MetaBitset`.
/// \tparam Values  Integral or enum values to represent in the resulting
/// `MetaBitset`.
///
/// \note The specified values will be cast to `PosT`, so choose `PosT` with
/// the potential for truncation / sign-extension in mind.
///
/// This type alias should be used in contexts where the min and max values are
/// already known in the parent context to avoid unnecessary duplication of
/// computation.
///
/// \warning Min and max bouds are not validated. This type alias saves
/// compile-time in contexts where the min and max are already computed.
template <typename PosT, PosT MinVal, PosT MaxVal, auto... Values>
using MakeMetaBitsetDetailed = typename metaset_detail::MakeMetaBitsetImpl<
    PosT, MinVal, MaxVal, static_cast<PosT>(Values)...>::type;

namespace metaset_detail {

template <typename PosT, auto... Values> struct MakeMetaBitsetFromValues;

/// Emtpy value set specialization.
template <typename PosT> struct MakeMetaBitsetFromValues<PosT> {
  using type = MetaBitset<PosT, 0>;
};

/// At least one value specialization.
template <typename PosT, auto Value0, auto... ValueN>
struct MakeMetaBitsetFromValues<PosT, Value0, ValueN...> {
private:
  static constexpr PosT Min = ce_min<PosT>(Value0, ValueN...);
  static constexpr PosT Max = ce_max<PosT>(Value0, ValueN...);

  static_assert(MetaBitsetNumWordsDetailed<PosT, Min, Max> <
                    MetaBitsetWordLimit,
                "MetaBitset exceeds word limit (llvm::MetaBitsetWordLimit).");

public:
  using type = MakeMetaBitsetDetailed<PosT, Min, Max, Value0, ValueN...>;
};

} // namespace metaset_detail

/// Create a `MetaBitset` containing the specified values.
///
/// \tparam PosT  Bit-position type to apply to the `MetaBitset`.
/// \tparam Values  Integral or enum values to represent in the resulting
/// `MetaBitset`.
///
/// \note The specified values will be cast to `PosT`, so choose this type with
/// the potential for truncation / sign-extension in mind.
///
/// The max and min of the specified values will be computed in order to
/// determine the size and offset of the resulting `MetaBitset`.  If this
/// information is already available, use `MakeMetaBitsetDetailed`.
template <typename PosT, auto... Values>
using MakeMetaBitset =
    typename metaset_detail::MakeMetaBitsetFromValues<PosT, Values...>::type;

namespace metaset_detail {

template <typename SeqT> class MakeMetaBitsetFromSequenceImpl {};

template <typename PosT, PosT... Values>
class MakeMetaBitsetFromSequenceImpl<std::integer_sequence<PosT, Values...>> {
public:
  using type = typename MakeMetaBitsetFromValues<PosT, Values...>::type;
};

template <typename SortedSeqT> class MakeMetaBitsetFromSortedSequenceImpl {};

/// Empty integer sequence specialization.
template <typename PosT>
class MakeMetaBitsetFromSortedSequenceImpl<std::integer_sequence<PosT>> {
public:
  using type = MetaBitset<PosT, 0>;
};

/// One or more values in the integer sequence specialization.
template <typename PosT, PosT Val0, PosT... ValN>
class MakeMetaBitsetFromSortedSequenceImpl<
    std::integer_sequence<PosT, Val0, ValN...>> {
  // The values in the sequence are sorted, so the first and last are min
  // and max respectively.
  static constexpr PosT MinVal = Val0;
  // A comma-expression evaluates to the sub-expression after the last comma.
  static constexpr PosT MaxVal = (Val0, ..., ValN);

public:
  using type = MakeMetaBitsetDetailed<PosT, MinVal, MaxVal, Val0, ValN...>;
};

} // namespace metaset_detail

/// Create a `MetaBitset` containing values taken from a
/// `std::integer_sequence`.
/// \relates MetaBitset
///
/// \tparam SeqT  `std::integer_sequence` of values to add to the generated
/// `MetaBitset`.
template <typename SeqT>
using MakeMetaBitsetFromSequence =
    typename metaset_detail::MakeMetaBitsetFromSequenceImpl<SeqT>::type;

/// Create a `MetaBitset` containing values taken from a sorted
/// `std::integer_sequence`.
/// \relates MetaBitset
///
/// \tparam SortedSeqT  Pre-sorted `std::integer_sequence` of values to add to
/// the generated `MetaBitset` type.
template <typename SortedSeqT>
using MakeMetaBitsetFromSortedSequence =
    typename metaset_detail::MakeMetaBitsetFromSortedSequenceImpl<
        SortedSeqT>::type;

template <typename SeqT> class MetaSequenceSet {};

/// Represent a set of values with a `std::integer_sequence`.
///
/// The elements of the sequence need not be sorted.
template <typename T, T... Values>
class MetaSequenceSet<std::integer_sequence<T, Values...>>
    : public MetaSequenceSetTag {
public:
  using value_type = T;
  using sequence_type = std::integer_sequence<T, Values...>;
  using sorted_sequence_type = SortSequence<sequence_type>;

public:
  constexpr MetaSequenceSet() {}

  /// Determine whether \p Val is equivalent to an element in the sequence.
  ///
  /// \return `true` if \p Val is in the set; `false` otherwise.
  ///
  /// \note Complexity = `O(n)` in sequence length.
  template <typename U> static constexpr bool contains(U const &Val) {
    if constexpr (sizeof...(Values) == 0)
      return false;
    else {
      T CastVal(static_cast<T>(Val));
      return (... || (CastVal == Values));
    }
  }

  static constexpr std::size_t count() { return sequence_type::size(); }

  static constexpr bool empty() { return sequence_type::size() == 0; }

  static constexpr auto to_array() // NOLINT (readability-identifier-naming)
  {
    return llvm::to_array(sequence_type());
  }

  static constexpr auto
  to_sorted_array() // NOLINT (readability-identifier-naming)
  {
    auto Arr = to_array();
    ce_sort_inplace(Arr);
    return Arr;
  }
};

namespace metaset_detail {

template <typename... MetaSetN> struct ToArrayImpl {};

template <typename SeqT, typename... MetaBitsetN>
struct ToArrayImpl<MetaSequenceSet<SeqT>, MetaBitsetN...> {};

} // namespace metaset_detail

/// Build a `MetaSequenceSet` from a list of integral or enum literals.
///
/// \tparam T  Integral or enum type of the sequence set to generate.
/// \tparam Values  Literals to add to the sequence set.
template <typename T, auto... Values>
using MakeMetaSequenceSet =
    MetaSequenceSet<std::integer_sequence<T, T(Values)...>>;

/// A meta-programming set composed from some number of `MetaBitset`s and
/// possibly one `MetaSequenceSet`.
///
/// The `MetaSparseSet` is created from a sequence of literal integral or enum
/// values using the helper type aliases.  See `MakeMetaSparseSet` and
/// `MakeMetaSparseSetFromSequence`.
///
/// If created using the helper type aliases, the parititoning algorithm is as
/// follows:
/// \code
///  // Determine the maximal slice length of `SortedSeq` starting at
///  // `StartIdx` which can be represented by a `MetaBitset` under the max word
///  // length.
///  //
///  // If the slice length is too sort, then return 1 to indicate a singleton
///  // value at `StartIdx`.
///  getSliceLen(SortedSeq, StartIdx) -> Integer
///
///  // Parition a sequence of values into clusters / slices (MetaBitsets) and
///  // potentially a set of singleton values (MetaSequenceSet).
///  partition(Values...):
///    SortedSeq = sort(Value...)
///    ValueIndex = 0
///    while ValueIndex < SortedSequence.size():
///      Len = getSliceLen(SortedSeq, ValueIndex)
///      if Len == 1:
///        // Slice is too small, more efficiently represented as a member of a
///        // MetaSequenceSet.
///        Singletons.push_back(SortedSeq[])
///      else:
///        // Record a slice to represent the current cluster of values.
///        Slices.push_back({ValueIndex, Len})
///      ValueIndex += Len
/// \endcode
///
/// The resulting `MetaSparseBitset` will have its slices / clusters represented
/// as disjoint `MetaBitset`s in ascending order of start offset followed by a
/// set of singletons represented by a single `MetaSequenceSet`.
///
/// Both the clusters and the trailing set of singletons are optional.  If all
/// values are represented by slices / clusters, then the trailing singletons
/// set will be omitted.  Conversely, if no values can be paritioned into
/// clusters then only the `MetaSequenceSet` will be present.
///
/// \note Elements of the trailing sequence set will always be sorted.
///
/// \todo Apply binary search to elements of the sorted trailing sequence set
/// when `contains()` is instantiated if the sequence is sufficiently large.
template <typename... MetaSetN>
class MetaSparseBitset : public MetaSparseBitsetTag {
  static_assert(sizeof...(MetaSetN) != 0);
  using MetaSet0 = PackElementT<0, MetaSetN...>;

public:
  using value_type = typename MetaSet0::value_type;

public:
  constexpr MetaSparseBitset() {}

  static constexpr std::size_t count() {
    return (MetaSetN::count() + ... + 0u);
  }

  static constexpr bool empty() {
    if constexpr (sizeof...(MetaSetN) == 1)
      // Expect the empty state of a MetaSparseBitset to be a single empty
      // MetaSequenceSet.
      return (MetaSetN::empty(), ...);
    else
      // The set is assumed to be non-empty if there are multiple sets in the
      // type pack.
      return false;
  }

  /// Determine whether \p Val is equivalent to an element in the set.
  ///
  /// \return `true` if \p Val is in the set; `false` otherwise.
  ///
  /// \note Complexity = `O(c + s)`; where c = the number of clusters and s =
  /// singleton sequence length.
  template <typename U> static constexpr bool contains(U const &Val) {
    return (... || MetaSetN::contains(Val));
  }

  static constexpr auto to_array() // NOLINT (readability-identifier-naming)
  {
    return toArrayImpl<count()>();
  }

  static constexpr auto
  to_sorted_array() // NOLINT (readability-identifier-naming)
  {
    return toSortedArrayImpl<count()>();
  }

private:
  /// Merge the sets in the order they appear in the type pack.
  template <std::size_t N>
  static constexpr std::array<value_type, N> toArrayImpl() {
    std::array<value_type, N> MergedArr{};

    auto FillFrom = [](auto Set, auto MergePos) constexpr {
      for (auto Val : Set.to_array())
        *MergePos++ = Val;
      return MergePos;
    };

    auto I = MergedArr.begin();
    ((I = FillFrom(MetaSetN(), I)), ...);

    return MergedArr;
  }

  template <std::size_t N>
  static constexpr std::array<value_type, N> toSortedArrayImpl() {
    using LastSetT = PackElementT<sizeof...(MetaSetN) - 1, MetaSetN...>;
    if constexpr (IsMetaSequenceSet<LastSetT>) {
      using MetaSeqT = LastSetT;
      std::array<value_type, N> MergedArr{};

      constexpr auto SeqArr = MetaSeqT::to_array();
      auto SeqPos = SeqArr.begin();
      auto SeqE = SeqArr.end();

      auto MergeFrom = [&](auto BitSet, auto MergePos) constexpr {
        using BitsetT =
            std::remove_cv_t<std::remove_reference_t<decltype(BitSet)>>;
        if constexpr (!std::is_same_v<MetaSeqT, BitsetT>) {
          for (auto Val : BitSet.to_array()) {
            while (SeqPos != SeqE && *SeqPos < Val)
              *MergePos++ = *SeqPos++;
            *MergePos++ = Val;
          }
        }
        return MergePos;
      };

      auto I = MergedArr.begin();
      ((I = MergeFrom(MetaSetN(), I)), ...);

      while (SeqPos != SeqE)
        *I++ = *SeqPos++;

      return MergedArr;
    } else {
      // The type pack is an ordered list of MetaBitsets; each containing
      // elements in sorted order.  A simple forward merge results in sorted
      // elements.
      return toArrayImpl<N>();
    }
  }

  template <std::size_t... I>
  static constexpr auto toSequenceImpl(std::index_sequence<I...> IdxSeq) {
    constexpr auto SeqArr = toArrayImpl<IdxSeq.size()>();
    return std::integer_sequence<value_type, SeqArr[I]...>();
  }

  template <std::size_t... I>
  static constexpr auto toSortedSequenceImpl(std::index_sequence<I...> IdxSeq) {
    constexpr auto SeqArr = toSortedArrayImpl<IdxSeq.size()>();
    return std::integer_sequence<value_type, SeqArr[I]...>();
  }

public:
  using sequence_type =
      decltype(toSequenceImpl(std::make_index_sequence<count()>()));
  using sorted_sequence_type =
      decltype(toSortedSequenceImpl(std::make_index_sequence<count()>()));
};

namespace metaset_detail {

template <typename SortedSeqT, size_t WordSize> class PartitionSparseBitset {
  using value_type = typename SortedSeqT::value_type;

  class Partitions {
    static constexpr size_t MinSliceLen = 3u;
    static constexpr std::array<value_type, SortedSeqT::size()> SeqArr =
        to_array(SortedSeqT());

    static constexpr size_t getSliceLen(size_t StartIdx) {
      auto Bottom = SeqArr[StartIdx];
      auto SeqStartPos = SeqArr.begin() + StartIdx;
      auto SeqCheckPos = SeqStartPos + 1u;
      auto SeqE = SeqArr.end();
      while (SeqCheckPos != SeqE && size_t(*SeqCheckPos - Bottom) < WordSize)
        ++SeqCheckPos;

      size_t Len = size_t(SeqCheckPos - SeqStartPos);

      // If slice is too short, return 1 to indicate a singleton at `StartIdx`.
      return Len < MinSliceLen ? 1u : Len;
    }

    struct CountsT {
      size_t NumSingles;
      size_t NumSlices;
    };

    static constexpr CountsT buildCounts() {
      if constexpr (SeqArr.size() == 0u)
        return {0u, 0u};

      size_t SliceStart = 0;
      size_t Singles = 0;
      size_t Slices = 0;
      while (SliceStart < SeqArr.size()) {
        size_t Len = getSliceLen(SliceStart);
        if (Len == 1u)
          ++Singles;
        else
          ++Slices;
        SliceStart += Len;
      }
      return {Singles, Slices};
    }

    static constexpr auto Counts = buildCounts();

  public:
    static constexpr size_t numSeqElems() { return Counts.NumSingles; }
    static constexpr size_t numSlices() { return Counts.NumSlices; }

  private:
    struct SliceT {
      size_t Start;
      size_t Length;
    };

    struct ParitionSpecT {
      std::array<value_type, numSeqElems()> Singles;
      std::array<SliceT, numSlices()> Slices;
    };

    static constexpr ParitionSpecT buildParitions() {
      ParitionSpecT PS{};

      size_t SliceIdx = 0u, SinglesIdx = 0u;
      size_t SliceStart = 0u;
      while (SliceStart < SeqArr.size()) {
        size_t Len = getSliceLen(SliceStart);
        if (Len == 1u)
          PS.Singles[SinglesIdx++] = SeqArr[SliceStart];
        else
          PS.Slices[SliceIdx++] = {SliceStart, Len};
        SliceStart += Len;
      }
      return PS;
    }

    static constexpr auto PartitionSpec = buildParitions();

  public:
    static constexpr size_t sliceStart(size_t Idx) {
      return PartitionSpec.Slices[Idx].Start;
    }

    static constexpr size_t sliceLen(size_t Idx) {
      return PartitionSpec.Slices[Idx].Length;
    }

    static constexpr value_type seqElem(size_t Idx) {
      return PartitionSpec.Singles[Idx];
    }
  };

  template <size_t SliceIdx>
  using MakeSliceBitset = MakeMetaBitsetFromSortedSequence<
      SliceSequence<Partitions::sliceStart(SliceIdx),
                    Partitions::sliceLen(SliceIdx), SortedSeqT>>;

  template <size_t... SeqIdx, size_t... SliceIdx>
  static constexpr auto partition(std::index_sequence<SeqIdx...>,
                                  std::index_sequence<SliceIdx...>) {

    if constexpr (Partitions::numSeqElems() == 0u)
      if constexpr (Partitions::numSlices() == 0u)
        // Empty input yields and empty sequence set.
        return MetaSparseBitset<MakeMetaSequenceSet<value_type>>();
      else
        // Avoid instantiating the singles sequence set if all values are
        // represented in bitsets.
        return MetaSparseBitset<MakeSliceBitset<SliceIdx>...>();
    else
      // CondGroupMetaSetParitioned looks for elements in order of the variadic
      // types. Put the sequence set last in hopes a value hits in one of the
      // bitsets since this would be cheaper at runtime.
      return MetaSparseBitset<
          MakeSliceBitset<SliceIdx>...,
          MakeMetaSequenceSet<value_type, Partitions::seqElem(SeqIdx)...>>();
  }

public:
  using type =
      decltype(partition(std::make_index_sequence<Partitions::numSeqElems()>(),
                         std::make_index_sequence<Partitions::numSlices()>()));
};

} // namespace metaset_detail

/// Create a `MetaSparseBitset` containing values taken from a
/// `std::integer_sequence`.
///
/// \tparam SeqT  `std::integer_sequence` of values to add to the generated
/// `MetaBitset`.
/// \tparam WordSize  Maximum number of bits allowed in a `MetaBitset` sub-word
/// within the sparse bitset.
///
/// \p SeqT may be an unsorted sequence.  While repeat elements are tolerated,
/// they do throw off heuristics and may result in an inefficient paritioning.
template <typename SeqT, size_t WordSize>
using MakeMetaSparseBitsetFromSequence =
    typename metaset_detail::PartitionSparseBitset<SortSequence<SeqT>,
                                                   WordSize>::type;

/// Create a `MetaSparseBitset` containing the specified values.
///
/// \tparam PosT  Integral or enum type used to represent the position / offset
/// in the bitset.
/// \tparam WordSize  Maximum number of bits allowed in a `MetaBitset` sub-word
/// within the sparse bitset.
/// \tparam Values  Values to represent in the resulting `MetaSparseBitset`.
///
/// \p Values may be an unsorted sequence.  While repeat elements are tolerated,
/// they do throw off heuristics and may result in an inefficient paritioning.
template <typename PosT, size_t WordSize, auto... Values>
using MakeMetaSparseBitset = typename metaset_detail::PartitionSparseBitset<
    SortLiterals<PosT, PosT(Values)...>, WordSize>::type;

/// Provide a container interface to a MetaSet type.
///
/// This class can be used as a mixin for any class accepting a `MetaSet` type
/// as a template parameter. For example:
/// \code{.cpp}
///  template <typename FooSetT>
///  class Foo : public MetaSetSortedContainer<FooSetT> {
///    static_assert(IsMetaSet<FooSetT>);
///  };
/// \endcode
template <typename MetaSetT> class MetaSetSortedContainer {
  static_assert(IsMetaSet<MetaSetT>);

  /// Use the instantiated type cache to ensure the array is only instantiated
  /// if one of the container-like member functions are called.
  template <typename T> struct ArrayInstantiator {
    static constexpr auto Array = T::to_sorted_array();
  };

public:
  using size_type = std::size_t;
  using value_type = typename MetaSetT::value_type;

private:
  using ArrayT = std::array<value_type, MetaSetT::count()>;

public:
  using reference = value_type const &;
  using const_reference = reference;
  using iterator = typename ArrayT::const_iterator;
  using const_iterator = iterator;
  using reverse_iterator = typename ArrayT::const_reverse_iterator;
  using const_reverse_iterator = reverse_iterator;

public:
  constexpr MetaSetSortedContainer() {}

  constexpr const_iterator begin() const {
    return ArrayInstantiator<MetaSetT>().Array.begin();
  }

  constexpr const_iterator end() const {
    return ArrayInstantiator<MetaSetT>().Array.end();
  }

  constexpr const_iterator cbegin() const {
    return ArrayInstantiator<MetaSetT>().Array.cbegin();
  }

  constexpr const_iterator cend() const {
    return ArrayInstantiator<MetaSetT>().Array.cend();
  }

  constexpr const_reverse_iterator rbegin() const {
    return ArrayInstantiator<MetaSetT>().Array.rbegin();
  }

  constexpr const_reverse_iterator rend() const {
    return ArrayInstantiator<MetaSetT>().Array.rend();
  }

  constexpr const_reverse_iterator crbegin() const {
    return ArrayInstantiator<MetaSetT>().Array.crbegin();
  }

  constexpr const_reverse_iterator crend() const {
    return ArrayInstantiator<MetaSetT>().Array.crend();
  }

  constexpr size_type size() const {
    return ArrayInstantiator<MetaSetT>().Array.size();
  }

  constexpr const_reference operator[](size_type Idx) const {
    static_assert(!MetaSetT::empty());
    return ArrayInstantiator<MetaSetT>().Array[Idx];
  }

  constexpr const_reference front() const {
    static_assert(!MetaSetT::empty());
    return ArrayInstantiator<MetaSetT>().Array.front();
  }

  constexpr const_reference back() const {
    static_assert(!MetaSetT::empty());
    return ArrayInstantiator<MetaSetT>().Array.back();
  }

  constexpr bool empty() const { return MetaSetT::empty(); }
};

namespace metaset_detail {

/// Count the number of increments applied; drop any assignments to the
/// dereferenced type.
///
/// This fake iterator is used by the set operation algorithms to determine the
/// required size of the array to generate at compile-time.
class CountIterator {
  std::size_t Count = 0;

  struct FakeReference {
    template <typename T> constexpr FakeReference &operator=(T const &) {
      return *this;
    }
  };

public:
  constexpr CountIterator() {}

  constexpr FakeReference operator*() { return {}; }

  constexpr CountIterator &operator++() {
    ++Count;
    return *this;
  }

  constexpr CountIterator operator++(int) {
    CountIterator Ret(*this);
    operator++();
    return Ret;
  }

  constexpr std::size_t count() const { return Count; }
};

template <typename Algorithm, std::size_t WordSize> class SetOperation {
  using value_type = typename Algorithm::value_type;
  using LeftSetT = typename Algorithm::LeftSetT;
  using RightSetT = typename Algorithm::RightSetT;

  static constexpr std::size_t ResultSize =
      Algorithm()(metaset_detail::CountIterator()).count();

  static constexpr auto makeResultArr() {
    std::array<value_type, ResultSize> ResultArr{};
    Algorithm()(ResultArr.begin());
    return ResultArr;
  }

  template <std::size_t... Idx>
  static constexpr auto makeResultSequence(std::index_sequence<Idx...>) {
    constexpr auto ResultArr = makeResultArr();
    return std::integer_sequence<value_type, ResultArr[Idx]...>();
  }

  using ResultSequence =
      decltype(makeResultSequence(std::make_index_sequence<ResultSize>()));

public:
  using type = typename PartitionSparseBitset<ResultSequence, WordSize>::type;
};

template <typename L, typename R> struct AlgoBase {
  using value_type = typename L::value_type;
  static_assert(
      std::is_same_v<value_type, typename R::value_type>,
      "MetaSet operations are only valid on sets with the same `value_type`.");

  using LeftSetT = L;
  using RightSetT = R;

  static constexpr auto ArrL = L::to_sorted_array();
  static constexpr auto ArrR = R::to_sorted_array();
};

template <typename L, typename R> struct UnionAlgo : AlgoBase<L, R> {
  using UnionAlgo::AlgoBase::ArrL;
  using UnionAlgo::AlgoBase::ArrR;

  template <typename OutIt> constexpr OutIt operator()(OutIt Out) const {
    auto I0 = ArrL.begin(), E0 = ArrL.end();
    auto I1 = ArrR.begin(), E1 = ArrR.end();

    while (I0 != E0) {
      if (I1 != E1) {
        if (*I0 == *I1) {
          *Out++ = *I0++;
          ++I1;
        } else if (*I0 < *I1) {
          *Out++ = *I0++;
        } else {
          *Out++ = *I1++;
        }
      } else {
        *Out++ = *I0++;
      }
    }

    while (I1 != E1)
      *Out++ = *I1++;

    return Out;
  }
};

template <typename L, typename R> struct IntersectionAlgo : AlgoBase<L, R> {
  using IntersectionAlgo::AlgoBase::ArrL;
  using IntersectionAlgo::AlgoBase::ArrR;

  template <typename OutIt> constexpr OutIt operator()(OutIt Out) const {
    auto I0 = ArrL.begin(), E0 = ArrL.end();
    auto I1 = ArrR.begin(), E1 = ArrR.end();

    while (I0 != E0 && I1 != E1) {
      if (*I0 == *I1) {
        *Out++ = *I0++;
        ++I1;
      } else if (*I0 < *I1) {
        ++I0;
      } else {
        ++I1;
      }
    }

    return Out;
  }
};

template <typename L, typename R> struct MinusAlgo : AlgoBase<L, R> {
  using MinusAlgo::AlgoBase::ArrL;
  using MinusAlgo::AlgoBase::ArrR;

  template <typename OutIt> constexpr OutIt operator()(OutIt Out) const {
    auto I0 = ArrL.begin(), E0 = ArrL.end();
    auto I1 = ArrR.begin(), E1 = ArrR.end();

    while (I0 != E0 && I1 != E1) {
      if (*I0 == *I1) {
        ++I0;
        ++I1;
      } else if (*I0 < *I1) {
        *Out++ = *I0++;
      } else {
        ++I1;
      }
    }

    // All remaining elements in I0's range do not exist in I1's range.
    while (I0 != E0)
      *Out++ = *I0++;

    return Out;
  }
};

inline constexpr std::size_t SetOpDefaultWordSize = 512;

} // namespace metaset_detail

/// Compute type resulting from a union of two meta set types.
///
/// The resulting type will always be expressed as a `MetaSparseBitset`.
///
/// \tparam L  Left-hand type to compute the union of.
/// \tparam R  Right-hand type to compute the union of.
/// \tparam WordSize  Sub-word size limit applied to the resulting
/// `MetaSparseBitset`.
///
/// Example usage:
/// \code{.cpp}
///  using EvensT = MakeMetaBitset<int, 2, 4, 6, 8, 10>;
///  using OddsT = MakeMetaBitset<int, 1, 3, 5, 7, 9>;
///
///  // Compute the union of `EvensT` and `OddsT`, limiting bitset word-size to
///  // 128 bits in the output `MetaSparseBitset`.
///  using Union = MetaSetUnion<EvensT, OddsT, 128>;
/// \endcode
template <typename L, typename R,
          std::size_t WordSize = metaset_detail::SetOpDefaultWordSize>
using MetaSetUnion =
    typename metaset_detail::SetOperation<metaset_detail::UnionAlgo<L, R>,
                                          WordSize>::type;

/// Compute the type resulting from an intersection of two meta set types.
///
/// The resulting type will always be expressed as a `MetaSparseBitset`.
///
/// \tparam L  Left-hand type to compute the intersection of.
/// \tparam R  Right-hand type to compute the intersection of.
/// \tparam WordSize  Sub-word size limit applied to the resulting
/// `MetaSparseBitset`.
///
/// Example usage:
/// \code{.cpp}
///  using EvensT = MakeMetaBitset<int, 2, 4, 6, 8, 10>;
///  using OneToFiveT = MakeMetaBitset<int, 1, 2, 3, 4, 5>;
///
///  // Compute the intersection of `EvensT` and `OneToFiveT`, limiting bitset
///  // word-size to 128 bits in the output `MetaSparseBitset`.
///  using Intersection = MetaSetIntersection<EvensT, OneToFiveT, 128>;
/// \endcode
template <typename L, typename R,
          std::size_t WordSize = metaset_detail::SetOpDefaultWordSize>
using MetaSetIntersection = typename metaset_detail::SetOperation<
    metaset_detail::IntersectionAlgo<L, R>, WordSize>::type;

/// Compute the type resulting from a subtraction of two meta set types.
///
/// The resulting type will always be expressed as a `MetaSparseBitset`.
///
/// \tparam L  Left-hand type to subtract from.
/// \tparam R  Right-hand type use as the subtrahend.
/// \tparam WordSize  Sub-word size limit applied to the resulting
/// `MetaSparseBitset`.
///
/// Example usage:
/// \code{.cpp}
///  using EvensT = MakeMetaBitset<int, 2, 4, 6, 8, 10>;
///  using OneToFiveT = MakeMetaBitset<int, 1, 2, 3, 4, 5>;
///
///  // Compute the difference by subtracting  `OneToFiveT` from `EvensT` and,
///  // limiting bitset word-size to 128 bits in the output `MetaSparseBitset`.
///  using Difference = MetaSetMinus<EvensT, OneToFiveT, 128>;
/// \endcode
template <typename L, typename R,
          std::size_t WordSize = metaset_detail::SetOpDefaultWordSize>
using MetaSetMinus =
    typename metaset_detail::SetOperation<metaset_detail::MinusAlgo<L, R>,
                                          WordSize>::type;

} // namespace llvm

#endif // LLVM_ADT_METASET_H
