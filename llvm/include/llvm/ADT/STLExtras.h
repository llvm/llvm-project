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

#include <cstddef> // for std::size_t
#include <cstdlib> // for qsort
#include <functional>
#include <iterator>
#include <utility> // for std::pair

namespace llvm {

//===----------------------------------------------------------------------===//
//     Extra additions to <functional>
//===----------------------------------------------------------------------===//

template<class Ty>
struct identity : public std::unary_function<Ty, Ty> {
  Ty &operator()(Ty &self) const {
    return self;
  }
  const Ty &operator()(const Ty &self) const {
    return self;
  }
};

template<class Ty>
struct less_ptr : public std::binary_function<Ty, Ty, bool> {
  bool operator()(const Ty* left, const Ty* right) const {
    return *left < *right;
  }
};

template<class Ty>
struct greater_ptr : public std::binary_function<Ty, Ty, bool> {
  bool operator()(const Ty* left, const Ty* right) const {
    return *right < *left;
  }
};

// deleter - Very very very simple method that is used to invoke operator
// delete on something.  It is used like this:
//
//   for_each(V.begin(), B.end(), deleter<Interval>);
//
template <class T>
inline void deleter(T *Ptr) {
  delete Ptr;
}



//===----------------------------------------------------------------------===//
//     Extra additions to <iterator>
//===----------------------------------------------------------------------===//

// mapped_iterator - This is a simple iterator adapter that causes a function to
// be dereferenced whenever operator* is invoked on the iterator.
//
template <class RootIt, class UnaryFunc>
class mapped_iterator {
  RootIt current;
  UnaryFunc Fn;
public:
  typedef typename std::iterator_traits<RootIt>::iterator_category
          iterator_category;
  typedef typename std::iterator_traits<RootIt>::difference_type
          difference_type;
  typedef typename UnaryFunc::result_type value_type;

  typedef void pointer;
  //typedef typename UnaryFunc::result_type *pointer;
  typedef void reference;        // Can't modify value returned by fn

  typedef RootIt iterator_type;
  typedef mapped_iterator<RootIt, UnaryFunc> _Self;

  inline const RootIt &getCurrent() const { return current; }
  inline const UnaryFunc &getFunc() const { return Fn; }

  inline explicit mapped_iterator(const RootIt &I, UnaryFunc F)
    : current(I), Fn(F) {}
  inline mapped_iterator(const mapped_iterator &It)
    : current(It.current), Fn(It.Fn) {}

  inline value_type operator*() const {   // All this work to do this
    return Fn(*current);         // little change
  }

  _Self& operator++() { ++current; return *this; }
  _Self& operator--() { --current; return *this; }
  _Self  operator++(int) { _Self __tmp = *this; ++current; return __tmp; }
  _Self  operator--(int) { _Self __tmp = *this; --current; return __tmp; }
  _Self  operator+    (difference_type n) const {
    return _Self(current + n, Fn);
  }
  _Self& operator+=   (difference_type n) { current += n; return *this; }
  _Self  operator-    (difference_type n) const {
    return _Self(current - n, Fn);
  }
  _Self& operator-=   (difference_type n) { current -= n; return *this; }
  reference operator[](difference_type n) const { return *(*this + n); }

  inline bool operator!=(const _Self &X) const { return !operator==(X); }
  inline bool operator==(const _Self &X) const { return current == X.current; }
  inline bool operator< (const _Self &X) const { return current <  X.current; }

  inline difference_type operator-(const _Self &X) const {
    return current - X.current;
  }
};

template <class _Iterator, class Func>
inline mapped_iterator<_Iterator, Func>
operator+(typename mapped_iterator<_Iterator, Func>::difference_type N,
          const mapped_iterator<_Iterator, Func>& X) {
  return mapped_iterator<_Iterator, Func>(X.getCurrent() - N, X.getFunc());
}


// map_iterator - Provide a convenient way to create mapped_iterators, just like
// make_pair is useful for creating pairs...
//
template <class ItTy, class FuncTy>
inline mapped_iterator<ItTy, FuncTy> map_iterator(const ItTy &I, FuncTy F) {
  return mapped_iterator<ItTy, FuncTy>(I, F);
}


// next/prior - These functions unlike std::advance do not modify the
// passed iterator but return a copy.
//
// next(myIt) returns copy of myIt incremented once
// next(myIt, n) returns copy of myIt incremented n times
// prior(myIt) returns copy of myIt decremented once
// prior(myIt, n) returns copy of myIt decremented n times

template <typename ItTy, typename Dist>
inline ItTy next(ItTy it, Dist n)
{
  std::advance(it, n);
  return it;
}

template <typename ItTy>
inline ItTy next(ItTy it)
{
  return ++it;
}

template <typename ItTy, typename Dist>
inline ItTy prior(ItTy it, Dist n)
{
  std::advance(it, -n);
  return it;
}

template <typename ItTy>
inline ItTy prior(ItTy it)
{
  return --it;
}

//===----------------------------------------------------------------------===//
//     Extra additions to <utility>
//===----------------------------------------------------------------------===//

// tie - this function ties two objects and returns a temporary object
// that is assignable from a std::pair. This can be used to make code
// more readable when using values returned from functions bundled in
// a std::pair. Since an example is worth 1000 words:
//
// typedef std::map<int, int> Int2IntMap;
//
// Int2IntMap myMap;
// Int2IntMap::iterator where;
// bool inserted;
// tie(where, inserted) = myMap.insert(std::make_pair(123,456));
//
// if (inserted)
//   // do stuff
// else
//   // do other stuff
template <typename T1, typename T2>
struct tier {
  typedef T1 &first_type;
  typedef T2 &second_type;

  first_type first;
  second_type second;

  tier(first_type f, second_type s) : first(f), second(s) { }
  tier& operator=(const std::pair<T1, T2>& p) {
    first = p.first;
    second = p.second;
    return *this;
  }
};

template <typename T1, typename T2>
inline tier<T1, T2> tie(T1& f, T2& s) {
  return tier<T1, T2>(f, s);
}

//===----------------------------------------------------------------------===//
//     Extra additions for arrays
//===----------------------------------------------------------------------===//

/// Find where an array ends (for ending iterators)
/// This returns a pointer to the byte immediately
/// after the end of an array.
template<class T, std::size_t N>
inline T *array_endof(T (&x)[N]) {
  return x+N;
}

/// Find the length of an array.
template<class T, std::size_t N>
inline size_t array_lengthof(T (&)[N]) {
  return N;
}

/// array_pod_sort_comparator - This is helper function for array_pod_sort,
/// which just uses operator< on T.
template<typename T>
inline int array_pod_sort_comparator(const void *P1, const void *P2) {
  if (*reinterpret_cast<const T*>(P1) < *reinterpret_cast<const T*>(P2))
    return -1;
  if (*reinterpret_cast<const T*>(P2) < *reinterpret_cast<const T*>(P1))
    return 1;
  return 0;
}

/// get_array_pad_sort_comparator - This is an internal helper function used to
/// get type deduction of T right.
template<typename T>
inline int (*get_array_pad_sort_comparator(const T &))
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
/// compared with operator< and can be moved with memcpy.  If this isn't true,
/// you should use std::sort.
///
/// NOTE: If qsort_r were portable, we could allow a custom comparator and
/// default to std::less.
template<class IteratorTy>
inline void array_pod_sort(IteratorTy Start, IteratorTy End) {
  // Don't dereference start iterator of empty sequence.
  if (Start == End) return;
  qsort(&*Start, End-Start, sizeof(*Start),
        get_array_pad_sort_comparator(*Start));
}

template<class IteratorTy>
inline void array_pod_sort(IteratorTy Start, IteratorTy End,
                                  int (*Compare)(const void*, const void*)) {
  // Don't dereference start iterator of empty sequence.
  if (Start == End) return;
  qsort(&*Start, End-Start, sizeof(*Start), Compare);
}

//===----------------------------------------------------------------------===//
//     Extra additions to <algorithm>
//===----------------------------------------------------------------------===//

/// For a container of pointers, deletes the pointers and then clears the
/// container.
template<typename Container>
void DeleteContainerPointers(Container &C) {
  for (typename Container::iterator I = C.begin(), E = C.end(); I != E; ++I)
    delete *I;
  C.clear();
}

/// In a container of pairs (usually a map) whose second element is a pointer,
/// deletes the second elements and then clears the container.
template<typename Container>
void DeleteContainerSeconds(Container &C) {
  for (typename Container::iterator I = C.begin(), E = C.end(); I != E; ++I)
    delete I->second;
  C.clear();
}

} // End llvm namespace

#endif
