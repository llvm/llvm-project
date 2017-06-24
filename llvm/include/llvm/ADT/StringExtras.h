//===- llvm/ADT/StringExtras.h - Useful string functions --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains some functions that are useful when dealing with strings.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STRINGEXTRAS_H
#define LLVM_ADT_STRINGEXTRAS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <utility>

namespace llvm {

template<typename T> class SmallVectorImpl;
class raw_ostream;

/// hexdigit - Return the hexadecimal character for the
/// given number \p X (which should be less than 16).
static inline char hexdigit(unsigned X, bool LowerCase = false) {
  const char HexChar = LowerCase ? 'a' : 'A';
  return X < 10 ? '0' + X : HexChar + X - 10;
}

/// Construct a string ref from a boolean.
static inline StringRef toStringRef(bool B) {
  return StringRef(B ? "true" : "false");
}

/// Construct a string ref from an array ref of unsigned chars.
static inline StringRef toStringRef(ArrayRef<uint8_t> Input) {
  return StringRef(reinterpret_cast<const char *>(Input.begin()), Input.size());
}

/// Interpret the given character \p C as a hexadecimal digit and return its
/// value.
///
/// If \p C is not a valid hex digit, -1U is returned.
static inline unsigned hexDigitValue(char C) {
  if (C >= '0' && C <= '9') return C-'0';
  if (C >= 'a' && C <= 'f') return C-'a'+10U;
  if (C >= 'A' && C <= 'F') return C-'A'+10U;
  return -1U;
}

static inline std::string utohexstr(uint64_t X, bool LowerCase = false) {
  char Buffer[17];
  char *BufPtr = std::end(Buffer);

  if (X == 0) *--BufPtr = '0';

  while (X) {
    unsigned char Mod = static_cast<unsigned char>(X) & 15;
    *--BufPtr = hexdigit(Mod, LowerCase);
    X >>= 4;
  }

  return std::string(BufPtr, std::end(Buffer));
}

/// Convert buffer \p Input to its hexadecimal representation.
/// The returned string is double the size of \p Input.
inline std::string toHex(StringRef Input) {
  static const char *const LUT = "0123456789ABCDEF";
  size_t Length = Input.size();

  std::string Output;
  Output.reserve(2 * Length);
  for (size_t i = 0; i < Length; ++i) {
    const unsigned char c = Input[i];
    Output.push_back(LUT[c >> 4]);
    Output.push_back(LUT[c & 15]);
  }
  return Output;
}

inline std::string toHex(ArrayRef<uint8_t> Input) {
  return toHex(toStringRef(Input));
}

static inline uint8_t hexFromNibbles(char MSB, char LSB) {
  unsigned U1 = hexDigitValue(MSB);
  unsigned U2 = hexDigitValue(LSB);
  assert(U1 != -1U && U2 != -1U);

  return static_cast<uint8_t>((U1 << 4) | U2);
}

/// Convert hexadecimal string \p Input to its binary representation.
/// The return string is half the size of \p Input.
static inline std::string fromHex(StringRef Input) {
  if (Input.empty())
    return std::string();

  std::string Output;
  Output.reserve((Input.size() + 1) / 2);
  if (Input.size() % 2 == 1) {
    Output.push_back(hexFromNibbles('0', Input.front()));
    Input = Input.drop_front();
  }

  assert(Input.size() % 2 == 0);
  while (!Input.empty()) {
    uint8_t Hex = hexFromNibbles(Input[0], Input[1]);
    Output.push_back(Hex);
    Input = Input.drop_front(2);
  }
  return Output;
}

/// \brief Convert the string \p S to an integer of the specified type using
/// the radix \p Base.  If \p Base is 0, auto-detects the radix.
/// Returns true if the number was successfully converted, false otherwise.
template <typename N> bool to_integer(StringRef S, N &Num, unsigned Base = 0) {
  return !S.getAsInteger(Base, Num);
}

namespace detail {
template <typename N>
inline bool to_float(const Twine &T, N &Num, N (*StrTo)(const char *, char **)) {
  SmallString<32> Storage;
  StringRef S = T.toNullTerminatedStringRef(Storage);
  char *End;
  N Temp = StrTo(S.data(), &End);
  if (*End != '\0')
    return false;
  Num = Temp;
  return true;
}
}

inline bool to_float(const Twine &T, float &Num) {
  return detail::to_float(T, Num, strtof);
}

inline bool to_float(const Twine &T, double &Num) {
  return detail::to_float(T, Num, strtod);
}

inline bool to_float(const Twine &T, long double &Num) {
  return detail::to_float(T, Num, strtold);
}

static inline std::string utostr(uint64_t X, bool isNeg = false) {
  char Buffer[21];
  char *BufPtr = std::end(Buffer);

  if (X == 0) *--BufPtr = '0';  // Handle special case...

  while (X) {
    *--BufPtr = '0' + char(X % 10);
    X /= 10;
  }

  if (isNeg) *--BufPtr = '-';   // Add negative sign...
  return std::string(BufPtr, std::end(Buffer));
}

static inline std::string itostr(int64_t X) {
  if (X < 0)
    return utostr(static_cast<uint64_t>(-X), true);
  else
    return utostr(static_cast<uint64_t>(X));
}

/// StrInStrNoCase - Portable version of strcasestr.  Locates the first
/// occurrence of string 's1' in string 's2', ignoring case.  Returns
/// the offset of s2 in s1 or npos if s2 cannot be found.
StringRef::size_type StrInStrNoCase(StringRef s1, StringRef s2);

/// getToken - This function extracts one token from source, ignoring any
/// leading characters that appear in the Delimiters string, and ending the
/// token at any of the characters that appear in the Delimiters string.  If
/// there are no tokens in the source string, an empty string is returned.
/// The function returns a pair containing the extracted token and the
/// remaining tail string.
std::pair<StringRef, StringRef> getToken(StringRef Source,
                                         StringRef Delimiters = " \t\n\v\f\r");

/// SplitString - Split up the specified string according to the specified
/// delimiters, appending the result fragments to the output list.
void SplitString(StringRef Source,
                 SmallVectorImpl<StringRef> &OutFragments,
                 StringRef Delimiters = " \t\n\v\f\r");

/// HashString - Hash function for strings.
///
/// This is the Bernstein hash function.
//
// FIXME: Investigate whether a modified bernstein hash function performs
// better: http://eternallyconfuzzled.com/tuts/algorithms/jsw_tut_hashing.aspx
//   X*33+c -> X*33^c
static inline unsigned HashString(StringRef Str, unsigned Result = 0) {
  for (StringRef::size_type i = 0, e = Str.size(); i != e; ++i)
    Result = Result * 33 + (unsigned char)Str[i];
  return Result;
}

/// Returns the English suffix for an ordinal integer (-st, -nd, -rd, -th).
static inline StringRef getOrdinalSuffix(unsigned Val) {
  // It is critically important that we do this perfectly for
  // user-written sequences with over 100 elements.
  switch (Val % 100) {
  case 11:
  case 12:
  case 13:
    return "th";
  default:
    switch (Val % 10) {
      case 1: return "st";
      case 2: return "nd";
      case 3: return "rd";
      default: return "th";
    }
  }
}

/// PrintEscapedString - Print each character of the specified string, escaping
/// it if it is not printable or if it is an escape char.
void PrintEscapedString(StringRef Name, raw_ostream &Out);

namespace detail {

template <typename IteratorT>
inline std::string join_impl(IteratorT Begin, IteratorT End,
                             StringRef Separator, std::input_iterator_tag) {
  std::string S;
  if (Begin == End)
    return S;

  S += (*Begin);
  while (++Begin != End) {
    S += Separator;
    S += (*Begin);
  }
  return S;
}

template <typename IteratorT>
inline std::string join_impl(IteratorT Begin, IteratorT End,
                             StringRef Separator, std::forward_iterator_tag) {
  std::string S;
  if (Begin == End)
    return S;

  size_t Len = (std::distance(Begin, End) - 1) * Separator.size();
  for (IteratorT I = Begin; I != End; ++I)
    Len += (*Begin).size();
  S.reserve(Len);
  S += (*Begin);
  while (++Begin != End) {
    S += Separator;
    S += (*Begin);
  }
  return S;
}

template <typename Sep>
inline void join_items_impl(std::string &Result, Sep Separator) {}

template <typename Sep, typename Arg>
inline void join_items_impl(std::string &Result, Sep Separator,
                            const Arg &Item) {
  Result += Item;
}

template <typename Sep, typename Arg1, typename... Args>
inline void join_items_impl(std::string &Result, Sep Separator, const Arg1 &A1,
                            Args &&... Items) {
  Result += A1;
  Result += Separator;
  join_items_impl(Result, Separator, std::forward<Args>(Items)...);
}

inline size_t join_one_item_size(char C) { return 1; }
inline size_t join_one_item_size(const char *S) { return S ? ::strlen(S) : 0; }

template <typename T> inline size_t join_one_item_size(const T &Str) {
  return Str.size();
}

inline size_t join_items_size() { return 0; }

template <typename A1> inline size_t join_items_size(const A1 &A) {
  return join_one_item_size(A);
}
template <typename A1, typename... Args>
inline size_t join_items_size(const A1 &A, Args &&... Items) {
  return join_one_item_size(A) + join_items_size(std::forward<Args>(Items)...);
}

} // end namespace detail

/// Joins the strings in the range [Begin, End), adding Separator between
/// the elements.
template <typename IteratorT>
inline std::string join(IteratorT Begin, IteratorT End, StringRef Separator) {
  using tag = typename std::iterator_traits<IteratorT>::iterator_category;
  return detail::join_impl(Begin, End, Separator, tag());
}

/// Joins the strings in the range [R.begin(), R.end()), adding Separator
/// between the elements.
template <typename Range>
inline std::string join(Range &&R, StringRef Separator) {
  return join(R.begin(), R.end(), Separator);
}

/// Joins the strings in the parameter pack \p Items, adding \p Separator
/// between the elements.  All arguments must be implicitly convertible to
/// std::string, or there should be an overload of std::string::operator+=()
/// that accepts the argument explicitly.
template <typename Sep, typename... Args>
inline std::string join_items(Sep Separator, Args &&... Items) {
  std::string Result;
  if (sizeof...(Items) == 0)
    return Result;

  size_t NS = detail::join_one_item_size(Separator);
  size_t NI = detail::join_items_size(std::forward<Args>(Items)...);
  Result.reserve(NI + (sizeof...(Items) - 1) * NS + 1);
  detail::join_items_impl(Result, Separator, std::forward<Args>(Items)...);
  return Result;
}

} // end namespace llvm

#endif // LLVM_ADT_STRINGEXTRAS_H
