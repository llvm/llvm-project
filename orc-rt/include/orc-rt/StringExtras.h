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
#include <charconv>
#include <cstdint>
#include <iterator>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

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
  [[maybe_unused]] const size_t PrevCapacity = Result.capacity();

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

/// A minimal, locale-free stand-in for std::ostringstream for building
/// diagnostic/error messages without dragging in <sstream> (and with it the
/// iostreams + locale machinery, which isn't freestanding-friendly). Integers
/// and pointers are formatted via std::to_chars, so there's no locale
/// dependence and no allocation beyond growing the result string.
class StringOutputStream {
public:
  template <typename T> struct HexFmt {
    T Value;
  };

  StringOutputStream &operator<<(char C) {
    S.push_back(C);
    return *this;
  }
  StringOutputStream &operator<<(bool B) {
    S.append(B ? "true" : "false");
    return *this;
  }
  StringOutputStream &operator<<(const char *Str) {
    S.append(Str);
    return *this;
  }
  StringOutputStream &operator<<(std::string_view Str) {
    S.append(Str);
    return *this;
  }

  StringOutputStream &operator<<(const void *P) {
    appendHex(reinterpret_cast<std::uintptr_t>(P));
    return *this;
  }

  template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
  StringOutputStream &operator<<(T Value) {
    // 3*sizeof(T) over-approximates decimal digits (~2.41/byte), + sign +
    // slack. Sized from the type so to_chars can never overflow, even for
    // extended integer types (e.g. __int128 under GNU extensions).
    char Buf[3 * sizeof(T) + 2];
    auto [Ptr, EC] = std::to_chars(Buf, std::end(Buf), Value);
    assert(EC == std::errc{} && "Buf outgrew its provable bound?");
    (void)EC;
    S.append(Buf, Ptr);
    return *this;
  }

  template <typename T> StringOutputStream &operator<<(HexFmt<T> H) {
    // hex() guarantees T is a non-bool integer, so make_unsigned_t is valid.
    appendHex(static_cast<std::make_unsigned_t<T>>(H.Value));
    return *this;
  }

  /// Access the accumulated string (mirrors std::ostringstream::str()).
  const std::string &str() const & { return S; }
  std::string str() && { return std::move(S); }

private:
  /// Append V as "0x"-prefixed lowercase hex. Leading zeros are dropped, so the
  /// printed width reflects the value, not the type.
  void appendHex(std::uintptr_t V) {
    char Buf[2 * sizeof(V)]; // Exact: 2 hex digits per byte, unsigned, no sign.
    auto [Ptr, EC] = std::to_chars(Buf, std::end(Buf), V, 16);
    assert(EC == std::errc{} && "hex buffer too small?");
    (void)EC;
    S.append("0x");
    S.append(Buf, Ptr);
  }

  std::string S;
};

/// Wrap an integer so StringOutputStream prints it as "0x"-prefixed lowercase
/// hex (formatted as its unsigned bit pattern). Pointers are already printed in
/// hex by the default operator<<, so hex() is for integers only.
template <typename T> StringOutputStream::HexFmt<T> hex(T Value) {
  static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>,
                "hex() is for integers; pointers already print in hex by "
                "default, and bool is not a numeric value");
  return {Value};
}

} // namespace orc_rt

#endif
