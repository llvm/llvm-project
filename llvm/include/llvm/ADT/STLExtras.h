//===- llvm/ADT/STLExtras.h - Useful STL related functions ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains some templates that are useful if you are working with the
// STL at all.
//
// No library is required when using these functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STLEXTRAS_H
#define LLVM_ADT_STLEXTRAS_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

namespace llvm {

// Only used by compiler if both template types are the same.  Useful when
// using SFINAE to test for the existence of member functions.
template <typename T, T> struct SameType;

namespace detail {

template <typename RangeT>
using IterOfRange = decltype(std::begin(std::declval<RangeT &>()));

template <typename RangeT>
using ValueOfRange = typename std::remove_reference<decltype(
    *std::begin(std::declval<RangeT &>()))>::type;

} // end namespace detail

//===----------------------------------------------------------------------===//
//     Extra additions to <functional>
//===----------------------------------------------------------------------===//

template <class Ty> struct identity {
  using argument_type = Ty;

  Ty &operator()(Ty &self) const {
    return self;
  }
  const Ty &operator()(const Ty &self) const {
    return self;
  }
};

template <class Ty> struct less_ptr {
  bool operator()(const Ty* left, const Ty* right) const {
    return *left < *right;
  }
};

template <class Ty> struct greater_ptr {
  bool operator()(const Ty* left, const Ty* right) const {
    return *right < *left;
  }
};

/// An efficient, type-erasing, non-owning reference to a callable. This is
/// intended for use as the type of a function parameter that is not used
/// after the function in question returns.
///
/// This class does not own the callable, so it is not in general safe to store
/// a function_ref.
template<typename Fn> class function_ref;

template<typename Ret, typename ...Params>
class function_ref<Ret(Params...)> {
  Ret (*callback)(intptr_t callable, Params ...params) = nullptr;
  intptr_t callable;

  template<typename Callable>
  static Ret callback_fn(intptr_t callable, Params ...params) {
    return (*reinterpret_cast<Callable*>(callable))(
        std::forward<Params>(params)...);
  }

public:
  function_ref() = default;

  template <typename Callable>
  function_ref(Callable &&callable,
               typename std::enable_if<
                   !std::is_same<typename std::remove_reference<Callable>::type,
                                 function_ref>::value>::type * = nullptr)
      : callback(callback_fn<typename std::remove_reference<Callable>::type>),
        callable(reinterpret_cast<intptr_t>(&callable)) {}

  Ret operator()(Params ...params) const {
    return callback(callable, std::forward<Params>(params)...);
  }

  operator bool() const { return callback; }
};

// deleter - Very very very simple method that is used to invoke operator
// delete on something.  It is used like this:
//
//   for_each(V.begin(), B.end(), deleter<Interval>);
template <class T>
inline void deleter(T *Ptr) {
  delete Ptr;
}

//===----------------------------------------------------------------------===//
//     Extra additions to <iterator>
//===----------------------------------------------------------------------===//

// mapped_iterator - This is a simple iterator adapter that causes a function to
// be applied whenever operator* is invoked on the iterator.
template <class RootIt, class UnaryFunc>
class mapped_iterator {
  RootIt current;
  UnaryFunc Fn;

public:
  using iterator_category =
      typename std::iterator_traits<RootIt>::iterator_category;
  using difference_type =
      typename std::iterator_traits<RootIt>::difference_type;
  using value_type =
      decltype(std::declval<UnaryFunc>()(*std::declval<RootIt>()));

  using pointer = void;
  using reference = void; // Can't modify value returned by fn

  using iterator_type = RootIt;

  inline explicit mapped_iterator(const RootIt &I, UnaryFunc F)
    : current(I), Fn(F) {}

  inline value_type operator*() const {   // All this work to do this
    return Fn(*current);         // little change
  }

  mapped_iterator &operator++() {
    ++current;
    return *this;
  }
  mapped_iterator &operator--() {
    --current;
    return *this;
  }
  mapped_iterator operator++(int) {
    mapped_iterator __tmp = *this;
    ++current;
    return __tmp;
  }
  mapped_iterator operator--(int) {
    mapped_iterator __tmp = *this;
    --current;
    return __tmp;
  }
  mapped_iterator operator+(difference_type n) const {
    return mapped_iterator(current + n, Fn);
  }
  mapped_iterator &operator+=(difference_type n) {
    current += n;
    return *this;
  }
  mapped_iterator operator-(difference_type n) const {
    return mapped_iterator(current - n, Fn);
  }
  mapped_iterator &operator-=(difference_type n) {
    current -= n;
    return *this;
  }
  reference operator[](difference_type n) const { return *(*this + n); }

  bool operator!=(const mapped_iterator &X) const { return !operator==(X); }
  bool operator==(const mapped_iterator &X) const {
    return current == X.current;
  }
  bool operator<(const mapped_iterator &X) const { return current < X.current; }

  difference_type operator-(const mapped_iterator &X) const {
    return current - X.current;
  }

  inline const RootIt &getCurrent() const { return current; }
  inline const UnaryFunc &getFunc() const { return Fn; }
};

template <class Iterator, class Func>
inline mapped_iterator<Iterator, Func>
operator+(typename mapped_iterator<Iterator, Func>::difference_type N,
          const mapped_iterator<Iterator, Func> &X) {
  return mapped_iterator<Iterator, Func>(X.getCurrent() - N, X.getFunc());
}

// map_iterator - Provide a convenient way to create mapped_iterators, just like
// make_pair is useful for creating pairs...
template <class ItTy, class FuncTy>
inline mapped_iterator<ItTy, FuncTy> map_iterator(const ItTy &I, FuncTy F) {
  return mapped_iterator<ItTy, FuncTy>(I, F);
}

/// Helper to determine if type T has a member called rbegin().
template <typename Ty> class has_rbegin_impl {
  using yes = char[1];
  using no = char[2];

  template <typename Inner>
  static yes& test(Inner *I, decltype(I->rbegin()) * = nullptr);

  template <typename>
  static no& test(...);

public:
  static const bool value = sizeof(test<Ty>(nullptr)) == sizeof(yes);
};

/// Metafunction to determine if T& or T has a member called rbegin().
template <typename Ty>
struct has_rbegin : has_rbegin_impl<typename std::remove_reference<Ty>::type> {
};

// Returns an iterator_range over the given container which iterates in reverse.
// Note that the container must have rbegin()/rend() methods for this to work.
template <typename ContainerTy>
auto reverse(ContainerTy &&C,
             typename std::enable_if<has_rbegin<ContainerTy>::value>::type * =
                 nullptr) -> decltype(make_range(C.rbegin(), C.rend())) {
  return make_range(C.rbegin(), C.rend());
}

// Returns a std::reverse_iterator wrapped around the given iterator.
template <typename IteratorTy>
std::reverse_iterator<IteratorTy> make_reverse_iterator(IteratorTy It) {
  return std::reverse_iterator<IteratorTy>(It);
}

// Returns an iterator_range over the given container which iterates in reverse.
// Note that the container must have begin()/end() methods which return
// bidirectional iterators for this to work.
template <typename ContainerTy>
auto reverse(
    ContainerTy &&C,
    typename std::enable_if<!has_rbegin<ContainerTy>::value>::type * = nullptr)
    -> decltype(make_range(llvm::make_reverse_iterator(std::end(C)),
                           llvm::make_reverse_iterator(std::begin(C)))) {
  return make_range(llvm::make_reverse_iterator(std::end(C)),
                    llvm::make_reverse_iterator(std::begin(C)));
}

/// An iterator adaptor that filters the elements of given inner iterators.
///
/// The predicate parameter should be a callable object that accepts the wrapped
/// iterator's reference type and returns a bool. When incrementing or
/// decrementing the iterator, it will call the predicate on each element and
/// skip any where it returns false.
///
/// \code
///   int A[] = { 1, 2, 3, 4 };
///   auto R = make_filter_range(A, [](int N) { return N % 2 == 1; });
///   // R contains { 1, 3 }.
/// \endcode
template <typename WrappedIteratorT, typename PredicateT>
class filter_iterator
    : public iterator_adaptor_base<
          filter_iterator<WrappedIteratorT, PredicateT>, WrappedIteratorT,
          typename std::common_type<
              std::forward_iterator_tag,
              typename std::iterator_traits<
                  WrappedIteratorT>::iterator_category>::type> {
  using BaseT = iterator_adaptor_base<
      filter_iterator<WrappedIteratorT, PredicateT>, WrappedIteratorT,
      typename std::common_type<
          std::forward_iterator_tag,
          typename std::iterator_traits<WrappedIteratorT>::iterator_category>::
          type>;

  struct PayloadType {
    WrappedIteratorT End;
    PredicateT Pred;
  };

  Optional<PayloadType> Payload;

  void findNextValid() {
    assert(Payload && "Payload should be engaged when findNextValid is called");
    while (this->I != Payload->End && !Payload->Pred(*this->I))
      BaseT::operator++();
  }

  // Construct the begin iterator. The begin iterator requires to know where end
  // is, so that it can properly stop when it hits end.
  filter_iterator(WrappedIteratorT Begin, WrappedIteratorT End, PredicateT Pred)
      : BaseT(std::move(Begin)),
        Payload(PayloadType{std::move(End), std::move(Pred)}) {
    findNextValid();
  }

  // Construct the end iterator. It's not incrementable, so Payload doesn't
  // have to be engaged.
  filter_iterator(WrappedIteratorT End) : BaseT(End) {}

public:
  using BaseT::operator++;

  filter_iterator &operator++() {
    BaseT::operator++();
    findNextValid();
    return *this;
  }

  template <typename RT, typename PT>
  friend iterator_range<filter_iterator<detail::IterOfRange<RT>, PT>>
  make_filter_range(RT &&, PT);
};

/// Convenience function that takes a range of elements and a predicate,
/// and return a new filter_iterator range.
///
/// FIXME: Currently if RangeT && is a rvalue reference to a temporary, the
/// lifetime of that temporary is not kept by the returned range object, and the
/// temporary is going to be dropped on the floor after the make_iterator_range
/// full expression that contains this function call.
template <typename RangeT, typename PredicateT>
iterator_range<filter_iterator<detail::IterOfRange<RangeT>, PredicateT>>
make_filter_range(RangeT &&Range, PredicateT Pred) {
  using FilterIteratorT =
      filter_iterator<detail::IterOfRange<RangeT>, PredicateT>;
  return make_range(FilterIteratorT(std::begin(std::forward<RangeT>(Range)),
                                    std::end(std::forward<RangeT>(Range)),
                                    std::move(Pred)),
                    FilterIteratorT(std::end(std::forward<RangeT>(Range))));
}

// forward declarations required by zip_shortest/zip_first
template <typename R, typename UnaryPredicate>
bool all_of(R &&range, UnaryPredicate P);

template <size_t... I> struct index_sequence;

template <class... Ts> struct index_sequence_for;

namespace detail {

using std::declval;

// We have to alias this since inlining the actual type at the usage site
// in the parameter list of iterator_facade_base<> below ICEs MSVC 2017.
template<typename... Iters> struct ZipTupleType {
  using type = std::tuple<decltype(*declval<Iters>())...>;
};

template <typename ZipType, typename... Iters>
using zip_traits = iterator_facade_base<
    ZipType, typename std::common_type<std::bidirectional_iterator_tag,
                                       typename std::iterator_traits<
                                           Iters>::iterator_category...>::type,
    // ^ TODO: Implement random access methods.
    typename ZipTupleType<Iters...>::type,
    typename std::iterator_traits<typename std::tuple_element<
        0, std::tuple<Iters...>>::type>::difference_type,
    // ^ FIXME: This follows boost::make_zip_iterator's assumption that all
    // inner iterators have the same difference_type. It would fail if, for
    // instance, the second field's difference_type were non-numeric while the
    // first is.
    typename ZipTupleType<Iters...>::type *,
    typename ZipTupleType<Iters...>::type>;

template <typename ZipType, typename... Iters>
struct zip_common : public zip_traits<ZipType, Iters...> {
  using Base = zip_traits<ZipType, Iters...>;
  using value_type = typename Base::value_type;

  std::tuple<Iters...> iterators;

protected:
  template <size_t... Ns> value_type deref(index_sequence<Ns...>) const {
    return value_type(*std::get<Ns>(iterators)...);
  }

  template <size_t... Ns>
  decltype(iterators) tup_inc(index_sequence<Ns...>) const {
    return std::tuple<Iters...>(std::next(std::get<Ns>(iterators))...);
  }

  template <size_t... Ns>
  decltype(iterators) tup_dec(index_sequence<Ns...>) const {
    return std::tuple<Iters...>(std::prev(std::get<Ns>(iterators))...);
  }

public:
  zip_common(Iters &&... ts) : iterators(std::forward<Iters>(ts)...) {}

  value_type operator*() { return deref(index_sequence_for<Iters...>{}); }

  const value_type operator*() const {
    return deref(index_sequence_for<Iters...>{});
  }

  ZipType &operator++() {
    iterators = tup_inc(index_sequence_for<Iters...>{});
    return *reinterpret_cast<ZipType *>(this);
  }

  ZipType &operator--() {
    static_assert(Base::IsBidirectional,
                  "All inner iterators must be at least bidirectional.");
    iterators = tup_dec(index_sequence_for<Iters...>{});
    return *reinterpret_cast<ZipType *>(this);
  }
};

template <typename... Iters>
struct zip_first : public zip_common<zip_first<Iters...>, Iters...> {
  using Base = zip_common<zip_first<Iters...>, Iters...>;

  bool operator==(const zip_first<Iters...> &other) const {
    return std::get<0>(this->iterators) == std::get<0>(other.iterators);
  }

  zip_first(Iters &&... ts) : Base(std::forward<Iters>(ts)...) {}
};

template <typename... Iters>
class zip_shortest : public zip_common<zip_shortest<Iters...>, Iters...> {
  template <size_t... Ns>
  bool test(const zip_shortest<Iters...> &other, index_sequence<Ns...>) const {
    return all_of(std::initializer_list<bool>{std::get<Ns>(this->iterators) !=
                                              std::get<Ns>(other.iterators)...},
                  identity<bool>{});
  }

public:
  using Base = zip_common<zip_shortest<Iters...>, Iters...>;

  zip_shortest(Iters &&... ts) : Base(std::forward<Iters>(ts)...) {}

  bool operator==(const zip_shortest<Iters...> &other) const {
    return !test(other, index_sequence_for<Iters...>{});
  }
};

template <template <typename...> class ItType, typename... Args> class zippy {
public:
  using iterator = ItType<decltype(std::begin(std::declval<Args>()))...>;
  using iterator_category = typename iterator::iterator_category;
  using value_type = typename iterator::value_type;
  using difference_type = typename iterator::difference_type;
  using pointer = typename iterator::pointer;
  using reference = typename iterator::reference;

private:
  std::tuple<Args...> ts;

  template <size_t... Ns> iterator begin_impl(index_sequence<Ns...>) const {
    return iterator(std::begin(std::get<Ns>(ts))...);
  }
  template <size_t... Ns> iterator end_impl(index_sequence<Ns...>) const {
    return iterator(std::end(std::get<Ns>(ts))...);
  }

public:
  zippy(Args &&... ts_) : ts(std::forward<Args>(ts_)...) {}

  iterator begin() const { return begin_impl(index_sequence_for<Args...>{}); }
  iterator end() const { return end_impl(index_sequence_for<Args...>{}); }
};

} // end namespace detail

/// zip iterator for two or more iteratable types.
template <typename T, typename U, typename... Args>
detail::zippy<detail::zip_shortest, T, U, Args...> zip(T &&t, U &&u,
                                                       Args &&... args) {
  return detail::zippy<detail::zip_shortest, T, U, Args...>(
      std::forward<T>(t), std::forward<U>(u), std::forward<Args>(args)...);
}

/// zip iterator that, for the sake of efficiency, assumes the first iteratee to
/// be the shortest.
template <typename T, typename U, typename... Args>
detail::zippy<detail::zip_first, T, U, Args...> zip_first(T &&t, U &&u,
                                                          Args &&... args) {
  return detail::zippy<detail::zip_first, T, U, Args...>(
      std::forward<T>(t), std::forward<U>(u), std::forward<Args>(args)...);
}

/// Iterator wrapper that concatenates sequences together.
///
/// This can concatenate different iterators, even with different types, into
/// a single iterator provided the value types of all the concatenated
/// iterators expose `reference` and `pointer` types that can be converted to
/// `ValueT &` and `ValueT *` respectively. It doesn't support more
/// interesting/customized pointer or reference types.
///
/// Currently this only supports forward or higher iterator categories as
/// inputs and always exposes a forward iterator interface.
template <typename ValueT, typename... IterTs>
class concat_iterator
    : public iterator_facade_base<concat_iterator<ValueT, IterTs...>,
                                  std::forward_iterator_tag, ValueT> {
  using BaseT = typename concat_iterator::iterator_facade_base;

  /// We store both the current and end iterators for each concatenated
  /// sequence in a tuple of pairs.
  ///
  /// Note that something like iterator_range seems nice at first here, but the
  /// range properties are of little benefit and end up getting in the way
  /// because we need to do mutation on the current iterators.
  std::tuple<std::pair<IterTs, IterTs>...> IterPairs;

  /// Attempts to increment a specific iterator.
  ///
  /// Returns true if it was able to increment the iterator. Returns false if
  /// the iterator is already at the end iterator.
  template <size_t Index> bool incrementHelper() {
    auto &IterPair = std::get<Index>(IterPairs);
    if (IterPair.first == IterPair.second)
      return false;

    ++IterPair.first;
    return true;
  }

  /// Increments the first non-end iterator.
  ///
  /// It is an error to call this with all iterators at the end.
  template <size_t... Ns> void increment(index_sequence<Ns...>) {
    // Build a sequence of functions to increment each iterator if possible.
    bool (concat_iterator::*IncrementHelperFns[])() = {
        &concat_iterator::incrementHelper<Ns>...};

    // Loop over them, and stop as soon as we succeed at incrementing one.
    for (auto &IncrementHelperFn : IncrementHelperFns)
      if ((this->*IncrementHelperFn)())
        return;

    llvm_unreachable("Attempted to increment an end concat iterator!");
  }

  /// Returns null if the specified iterator is at the end. Otherwise,
  /// dereferences the iterator and returns the address of the resulting
  /// reference.
  template <size_t Index> ValueT *getHelper() const {
    auto &IterPair = std::get<Index>(IterPairs);
    if (IterPair.first == IterPair.second)
      return nullptr;

    return &*IterPair.first;
  }

  /// Finds the first non-end iterator, dereferences, and returns the resulting
  /// reference.
  ///
  /// It is an error to call this with all iterators at the end.
  template <size_t... Ns> ValueT &get(index_sequence<Ns...>) const {
    // Build a sequence of functions to get from iterator if possible.
    ValueT *(concat_iterator::*GetHelperFns[])() const = {
        &concat_iterator::getHelper<Ns>...};

    // Loop over them, and return the first result we find.
    for (auto &GetHelperFn : GetHelperFns)
      if (ValueT *P = (this->*GetHelperFn)())
        return *P;

    llvm_unreachable("Attempted to get a pointer from an end concat iterator!");
  }

public:
  /// Constructs an iterator from a squence of ranges.
  ///
  /// We need the full range to know how to switch between each of the
  /// iterators.
  template <typename... RangeTs>
  explicit concat_iterator(RangeTs &&... Ranges)
      : IterPairs({std::begin(Ranges), std::end(Ranges)}...) {}

  using BaseT::operator++;

  concat_iterator &operator++() {
    increment(index_sequence_for<IterTs...>());
    return *this;
  }

  ValueT &operator*() const { return get(index_sequence_for<IterTs...>()); }

  bool operator==(const concat_iterator &RHS) const {
    return IterPairs == RHS.IterPairs;
  }
};

namespace detail {

/// Helper to store a sequence of ranges being concatenated and access them.
///
/// This is designed to facilitate providing actual storage when temporaries
/// are passed into the constructor such that we can use it as part of range
/// based for loops.
template <typename ValueT, typename... RangeTs> class concat_range {
public:
  using iterator =
      concat_iterator<ValueT,
                      decltype(std::begin(std::declval<RangeTs &>()))...>;

private:
  std::tuple<RangeTs...> Ranges;

  template <size_t... Ns> iterator begin_impl(index_sequence<Ns...>) {
    return iterator(std::get<Ns>(Ranges)...);
  }
  template <size_t... Ns> iterator end_impl(index_sequence<Ns...>) {
    return iterator(make_range(std::end(std::get<Ns>(Ranges)),
                               std::end(std::get<Ns>(Ranges)))...);
  }

public:
  concat_range(RangeTs &&... Ranges)
      : Ranges(std::forward<RangeTs>(Ranges)...) {}

  iterator begin() { return begin_impl(index_sequence_for<RangeTs...>{}); }
  iterator end() { return end_impl(index_sequence_for<RangeTs...>{}); }
};

} // end namespace detail

/// Concatenated range across two or more ranges.
///
/// The desired value type must be explicitly specified.
template <typename ValueT, typename... RangeTs>
detail::concat_range<ValueT, RangeTs...> concat(RangeTs &&... Ranges) {
  static_assert(sizeof...(RangeTs) > 1,
                "Need more than one range to concatenate!");
  return detail::concat_range<ValueT, RangeTs...>(
      std::forward<RangeTs>(Ranges)...);
}

//===----------------------------------------------------------------------===//
//     Extra additions to <utility>
//===----------------------------------------------------------------------===//

/// \brief Function object to check whether the first component of a std::pair
/// compares less than the first component of another std::pair.
struct less_first {
  template <typename T> bool operator()(const T &lhs, const T &rhs) const {
    return lhs.first < rhs.first;
  }
};

/// \brief Function object to check whether the second component of a std::pair
/// compares less than the second component of another std::pair.
struct less_second {
  template <typename T> bool operator()(const T &lhs, const T &rhs) const {
    return lhs.second < rhs.second;
  }
};

// A subset of N3658. More stuff can be added as-needed.

/// \brief Represents a compile-time sequence of integers.
template <class T, T... I> struct integer_sequence {
  using value_type = T;

  static constexpr size_t size() { return sizeof...(I); }
};

/// \brief Alias for the common case of a sequence of size_ts.
template <size_t... I>
struct index_sequence : integer_sequence<std::size_t, I...> {};

template <std::size_t N, std::size_t... I>
struct build_index_impl : build_index_impl<N - 1, N - 1, I...> {};
template <std::size_t... I>
struct build_index_impl<0, I...> : index_sequence<I...> {};

/// \brief Creates a compile-time integer sequence for a parameter pack.
template <class... Ts>
struct index_sequence_for : build_index_impl<sizeof...(Ts)> {};

/// Utility type to build an inheritance chain that makes it easy to rank
/// overload candidates.
template <int N> struct rank : rank<N - 1> {};
template <> struct rank<0> {};

/// \brief traits class for checking whether type T is one of any of the given
/// types in the variadic list.
template <typename T, typename... Ts> struct is_one_of {
  static const bool value = false;
};

template <typename T, typename U, typename... Ts>
struct is_one_of<T, U, Ts...> {
  static const bool value =
      std::is_same<T, U>::value || is_one_of<T, Ts...>::value;
};

/// \brief traits class for checking whether type T is a base class for all
///  the given types in the variadic list.
template <typename T, typename... Ts> struct are_base_of {
  static const bool value = true;
};

template <typename T, typename U, typename... Ts>
struct are_base_of<T, U, Ts...> {
  static const bool value =
      std::is_base_of<T, U>::value && are_base_of<T, Ts...>::value;
};

//===----------------------------------------------------------------------===//
//     Extra additions for arrays
//===----------------------------------------------------------------------===//

/// Find the length of an array.
template <class T, std::size_t N>
constexpr inline size_t array_lengthof(T (&)[N]) {
  return N;
}

/// Adapt std::less<T> for array_pod_sort.
template<typename T>
inline int array_pod_sort_comparator(const void *P1, const void *P2) {
  if (std::less<T>()(*reinterpret_cast<const T*>(P1),
                     *reinterpret_cast<const T*>(P2)))
    return -1;
  if (std::less<T>()(*reinterpret_cast<const T*>(P2),
                     *reinterpret_cast<const T*>(P1)))
    return 1;
  return 0;
}

/// get_array_pod_sort_comparator - This is an internal helper function used to
/// get type deduction of T right.
template<typename T>
inline int (*get_array_pod_sort_comparator(const T &))
             (const void*, const void*) {
  return array_pod_sort_comparator<T>;
}

/// array_pod_sort - This sorts an array with the specified start and end
/// extent.  This is just like std::sort, except that it calls qsort instead of
/// using an inlined template.  qsort is slightly slower than std::sort, but
/// most sorts are not performance critical in LLVM and std::sort has to be
/// template instantiated for each type, leading to significant measured code
/// bloat.  This function should generally be used instead of std::sort where
/// possible.
///
/// This function assumes that you have simple POD-like types that can be
/// compared with std::less and can be moved with memcpy.  If this isn't true,
/// you should use std::sort.
///
/// NOTE: If qsort_r were portable, we could allow a custom comparator and
/// default to std::less.
template<class IteratorTy>
inline void array_pod_sort(IteratorTy Start, IteratorTy End) {
  // Don't inefficiently call qsort with one element or trigger undefined
  // behavior with an empty sequence.
  auto NElts = End - Start;
  if (NElts <= 1) return;
  qsort(&*Start, NElts, sizeof(*Start), get_array_pod_sort_comparator(*Start));
}

template <class IteratorTy>
inline void array_pod_sort(
    IteratorTy Start, IteratorTy End,
    int (*Compare)(
        const typename std::iterator_traits<IteratorTy>::value_type *,
        const typename std::iterator_traits<IteratorTy>::value_type *)) {
  // Don't inefficiently call qsort with one element or trigger undefined
  // behavior with an empty sequence.
  auto NElts = End - Start;
  if (NElts <= 1) return;
  qsort(&*Start, NElts, sizeof(*Start),
        reinterpret_cast<int (*)(const void *, const void *)>(Compare));
}

//===----------------------------------------------------------------------===//
//     Extra additions to <algorithm>
//===----------------------------------------------------------------------===//

/// For a container of pointers, deletes the pointers and then clears the
/// container.
template<typename Container>
void DeleteContainerPointers(Container &C) {
  for (auto V : C)
    delete V;
  C.clear();
}

/// In a container of pairs (usually a map) whose second element is a pointer,
/// deletes the second elements and then clears the container.
template<typename Container>
void DeleteContainerSeconds(Container &C) {
  for (auto &V : C)
    delete V.second;
  C.clear();
}

/// Provide wrappers to std::all_of which take ranges instead of having to pass
/// begin/end explicitly.
template <typename R, typename UnaryPredicate>
bool all_of(R &&Range, UnaryPredicate P) {
  return std::all_of(std::begin(Range), std::end(Range), P);
}

/// Provide wrappers to std::any_of which take ranges instead of having to pass
/// begin/end explicitly.
template <typename R, typename UnaryPredicate>
bool any_of(R &&Range, UnaryPredicate P) {
  return std::any_of(std::begin(Range), std::end(Range), P);
}

/// Provide wrappers to std::none_of which take ranges instead of having to pass
/// begin/end explicitly.
template <typename R, typename UnaryPredicate>
bool none_of(R &&Range, UnaryPredicate P) {
  return std::none_of(std::begin(Range), std::end(Range), P);
}

/// Provide wrappers to std::find which take ranges instead of having to pass
/// begin/end explicitly.
template <typename R, typename T>
auto find(R &&Range, const T &Val) -> decltype(std::begin(Range)) {
  return std::find(std::begin(Range), std::end(Range), Val);
}

/// Provide wrappers to std::find_if which take ranges instead of having to pass
/// begin/end explicitly.
template <typename R, typename UnaryPredicate>
auto find_if(R &&Range, UnaryPredicate P) -> decltype(std::begin(Range)) {
  return std::find_if(std::begin(Range), std::end(Range), P);
}

template <typename R, typename UnaryPredicate>
auto find_if_not(R &&Range, UnaryPredicate P) -> decltype(std::begin(Range)) {
  return std::find_if_not(std::begin(Range), std::end(Range), P);
}

/// Provide wrappers to std::remove_if which take ranges instead of having to
/// pass begin/end explicitly.
template <typename R, typename UnaryPredicate>
auto remove_if(R &&Range, UnaryPredicate P) -> decltype(std::begin(Range)) {
  return std::remove_if(std::begin(Range), std::end(Range), P);
}

/// Provide wrappers to std::copy_if which take ranges instead of having to
/// pass begin/end explicitly.
template <typename R, typename OutputIt, typename UnaryPredicate>
OutputIt copy_if(R &&Range, OutputIt Out, UnaryPredicate P) {
  return std::copy_if(std::begin(Range), std::end(Range), Out, P);
}

/// Wrapper function around std::find to detect if an element exists
/// in a container.
template <typename R, typename E>
bool is_contained(R &&Range, const E &Element) {
  return std::find(std::begin(Range), std::end(Range), Element) !=
         std::end(Range);
}

/// Wrapper function around std::count to count the number of times an element
/// \p Element occurs in the given range \p Range.
template <typename R, typename E>
auto count(R &&Range, const E &Element) -> typename std::iterator_traits<
    decltype(std::begin(Range))>::difference_type {
  return std::count(std::begin(Range), std::end(Range), Element);
}

/// Wrapper function around std::count_if to count the number of times an
/// element satisfying a given predicate occurs in a range.
template <typename R, typename UnaryPredicate>
auto count_if(R &&Range, UnaryPredicate P) -> typename std::iterator_traits<
    decltype(std::begin(Range))>::difference_type {
  return std::count_if(std::begin(Range), std::end(Range), P);
}

/// Wrapper function around std::transform to apply a function to a range and
/// store the result elsewhere.
template <typename R, typename OutputIt, typename UnaryPredicate>
OutputIt transform(R &&Range, OutputIt d_first, UnaryPredicate P) {
  return std::transform(std::begin(Range), std::end(Range), d_first, P);
}

/// Provide wrappers to std::partition which take ranges instead of having to
/// pass begin/end explicitly.
template <typename R, typename UnaryPredicate>
auto partition(R &&Range, UnaryPredicate P) -> decltype(std::begin(Range)) {
  return std::partition(std::begin(Range), std::end(Range), P);
}

/// Provide wrappers to std::lower_bound which take ranges instead of having to
/// pass begin/end explicitly.
template <typename R, typename ForwardIt>
auto lower_bound(R &&Range, ForwardIt I) -> decltype(std::begin(Range)) {
  return std::lower_bound(std::begin(Range), std::end(Range), I);
}

/// \brief Given a range of type R, iterate the entire range and return a
/// SmallVector with elements of the vector.  This is useful, for example,
/// when you want to iterate a range and then sort the results.
template <unsigned Size, typename R>
SmallVector<typename std::remove_const<detail::ValueOfRange<R>>::type, Size>
to_vector(R &&Range) {
  return {std::begin(Range), std::end(Range)};
}

/// Provide a container algorithm similar to C++ Library Fundamentals v2's
/// `erase_if` which is equivalent to:
///
///   C.erase(remove_if(C, pred), C.end());
///
/// This version works for any container with an erase method call accepting
/// two iterators.
template <typename Container, typename UnaryPredicate>
void erase_if(Container &C, UnaryPredicate P) {
  C.erase(remove_if(C, P), C.end());
}

//===----------------------------------------------------------------------===//
//     Extra additions to <memory>
//===----------------------------------------------------------------------===//

// Implement make_unique according to N3656.

/// \brief Constructs a `new T()` with the given args and returns a
///        `unique_ptr<T>` which owns the object.
///
/// Example:
///
///     auto p = make_unique<int>();
///     auto p = make_unique<std::tuple<int, int>>(0, 1);
template <class T, class... Args>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Args &&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

/// \brief Constructs a `new T[n]` with the given args and returns a
///        `unique_ptr<T[]>` which owns the object.
///
/// \param n size of the new array.
///
/// Example:
///
///     auto p = make_unique<int[]>(2); // value-initializes the array with 0's.
template <class T>
typename std::enable_if<std::is_array<T>::value && std::extent<T>::value == 0,
                        std::unique_ptr<T>>::type
make_unique(size_t n) {
  return std::unique_ptr<T>(new typename std::remove_extent<T>::type[n]());
}

/// This function isn't used and is only here to provide better compile errors.
template <class T, class... Args>
typename std::enable_if<std::extent<T>::value != 0>::type
make_unique(Args &&...) = delete;

struct FreeDeleter {
  void operator()(void* v) {
    ::free(v);
  }
};

template<typename First, typename Second>
struct pair_hash {
  size_t operator()(const std::pair<First, Second> &P) const {
    return std::hash<First>()(P.first) * 31 + std::hash<Second>()(P.second);
  }
};

/// A functor like C++14's std::less<void> in its absence.
struct less {
  template <typename A, typename B> bool operator()(A &&a, B &&b) const {
    return std::forward<A>(a) < std::forward<B>(b);
  }
};

/// A functor like C++14's std::equal<void> in its absence.
struct equal {
  template <typename A, typename B> bool operator()(A &&a, B &&b) const {
    return std::forward<A>(a) == std::forward<B>(b);
  }
};

/// Binary functor that adapts to any other binary functor after dereferencing
/// operands.
template <typename T> struct deref {
  T func;

  // Could be further improved to cope with non-derivable functors and
  // non-binary functors (should be a variadic template member function
  // operator()).
  template <typename A, typename B>
  auto operator()(A &lhs, B &rhs) const -> decltype(func(*lhs, *rhs)) {
    assert(lhs);
    assert(rhs);
    return func(*lhs, *rhs);
  }
};

namespace detail {

template <typename R> class enumerator_iter;

template <typename R> struct result_pair {
  friend class enumerator_iter<R>;

  result_pair() = default;
  result_pair(std::size_t Index, IterOfRange<R> Iter)
      : Index(Index), Iter(Iter) {}

  result_pair<R> &operator=(const result_pair<R> &Other) {
    Index = Other.Index;
    Iter = Other.Iter;
    return *this;
  }

  std::size_t index() const { return Index; }
  const ValueOfRange<R> &value() const { return *Iter; }
  ValueOfRange<R> &value() { return *Iter; }

private:
  std::size_t Index = std::numeric_limits<std::size_t>::max();
  IterOfRange<R> Iter;
};

template <typename R>
class enumerator_iter
    : public iterator_facade_base<
          enumerator_iter<R>, std::forward_iterator_tag, result_pair<R>,
          typename std::iterator_traits<IterOfRange<R>>::difference_type,
          typename std::iterator_traits<IterOfRange<R>>::pointer,
          typename std::iterator_traits<IterOfRange<R>>::reference> {
  using result_type = result_pair<R>;

public:
  explicit enumerator_iter(IterOfRange<R> EndIter)
      : Result(std::numeric_limits<size_t>::max(), EndIter) {}

  enumerator_iter(std::size_t Index, IterOfRange<R> Iter)
      : Result(Index, Iter) {}

  result_type &operator*() { return Result; }
  const result_type &operator*() const { return Result; }

  enumerator_iter<R> &operator++() {
    assert(Result.Index != std::numeric_limits<size_t>::max());
    ++Result.Iter;
    ++Result.Index;
    return *this;
  }

  bool operator==(const enumerator_iter<R> &RHS) const {
    // Don't compare indices here, only iterators.  It's possible for an end
    // iterator to have different indices depending on whether it was created
    // by calling std::end() versus incrementing a valid iterator.
    return Result.Iter == RHS.Result.Iter;
  }

  enumerator_iter<R> &operator=(const enumerator_iter<R> &Other) {
    Result = Other.Result;
    return *this;
  }

private:
  result_type Result;
};

template <typename R> class enumerator {
public:
  explicit enumerator(R &&Range) : TheRange(std::forward<R>(Range)) {}

  enumerator_iter<R> begin() {
    return enumerator_iter<R>(0, std::begin(TheRange));
  }

  enumerator_iter<R> end() {
    return enumerator_iter<R>(std::end(TheRange));
  }

private:
  R TheRange;
};

} // end namespace detail

/// Given an input range, returns a new range whose values are are pair (A,B)
/// such that A is the 0-based index of the item in the sequence, and B is
/// the value from the original sequence.  Example:
///
/// std::vector<char> Items = {'A', 'B', 'C', 'D'};
/// for (auto X : enumerate(Items)) {
///   printf("Item %d - %c\n", X.index(), X.value());
/// }
///
/// Output:
///   Item 0 - A
///   Item 1 - B
///   Item 2 - C
///   Item 3 - D
///
template <typename R> detail::enumerator<R> enumerate(R &&TheRange) {
  return detail::enumerator<R>(std::forward<R>(TheRange));
}

namespace detail {

template <typename F, typename Tuple, std::size_t... I>
auto apply_tuple_impl(F &&f, Tuple &&t, index_sequence<I...>)
    -> decltype(std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...)) {
  return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
}

} // end namespace detail

/// Given an input tuple (a1, a2, ..., an), pass the arguments of the
/// tuple variadically to f as if by calling f(a1, a2, ..., an) and
/// return the result.
template <typename F, typename Tuple>
auto apply_tuple(F &&f, Tuple &&t) -> decltype(detail::apply_tuple_impl(
    std::forward<F>(f), std::forward<Tuple>(t),
    build_index_impl<
        std::tuple_size<typename std::decay<Tuple>::type>::value>{})) {
  using Indices = build_index_impl<
      std::tuple_size<typename std::decay<Tuple>::type>::value>;

  return detail::apply_tuple_impl(std::forward<F>(f), std::forward<Tuple>(t),
                                  Indices{});
}

} // end namespace llvm

#endif // LLVM_ADT_STLEXTRAS_H
