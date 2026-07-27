//===--- StringExtras.h - Stolen from llvm/ADT/StringExtras.h ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef ORC_RT_STRINGEXTRAS_H
#define ORC_RT_STRINGEXTRAS_H

#include <cassert>
#include <iterator>
#include <string>
#include <type_traits>
namespace orc_rt {
/// A simplification of what is in llvm/ADT/StringExtras.h
/// Preserves the behaviour but removes tag dispatch
/// Will assert if iterator is not a forward iterator.
template <typename IteratorT>
std::string join(IteratorT Begin, IteratorT End, std::string_view Separator) {
  using Category = typename std::iterator_traits<IteratorT>::iterator_category;
  static_assert(std::is_base_of_v<std::forward_iterator_tag, Category>,
                "join requires forward iterators (range is traversed twice)");
  if (Begin == End)
    return {};

  size_t Size = 0, Count = 0;
  for (IteratorT I = Begin; I != End; ++I, ++Count)
    Size += std::string_view(*I).size();

  std::string Result;
  Result.reserve(Size + (Count - 1) * Separator.size());
  const size_t PrevCapacity = Result.capacity();
  [maybe_used]] PrevCapacity;

  Result += std::string_view(*Begin);
  while (++Begin != End) {
    Result += Separator;
    Result += std::string_view(*Begin);
  }

  assert(PrevCapacity == Result.capacity() && "String grew during building");
  return Result;
}

template <typename Range>
std::string join(Range &&R, std::string_view Separator) {
  return join(std::begin(R), std::end(R), Separator);
}

inline std::string join(std::initializer_list<std::string_view> Elements,
                        std::string_view Separator) {
  return join(Elements.begin(), Elements.end(), Separator);
}
} // namespace orc_rt
#endif
