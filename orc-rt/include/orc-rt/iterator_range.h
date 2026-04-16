//===---- iterator_range.h -- Simple iterator range template ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple iterator range template.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_ITERATOR_RANGE_H
#define ORC_RT_ITERATOR_RANGE_H

#include <iterator>

namespace orc_rt {

/// A simple wrapper around a pair of iterators, enabling range-based for
/// loops over iterator pairs or subranges of containers.
template <typename IteratorT> class iterator_range {
public:
  /// Construct an iterator_range from a container or range. The underlying
  /// container must outlive this iterator_range.
  template <typename Container>
  iterator_range(Container &&C) : Begin(std::begin(C)), End(std::end(C)) {}

  /// Construct an iterator_range from an explicit begin/end pair.
  iterator_range(IteratorT Begin, IteratorT End)
      : Begin(std::move(Begin)), End(std::move(End)) {}

  IteratorT begin() const { return Begin; }
  IteratorT end() const { return End; }
  bool empty() const { return Begin == End; }

private:
  IteratorT Begin, End;
};

template <typename Container>
iterator_range(Container &&)
    -> iterator_range<decltype(std::begin(std::declval<Container &&>()))>;

} // namespace orc_rt

#endif // ORC_RT_ITERATOR_RANGE_H
