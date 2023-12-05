//===-- CharSet.h - Utility class to convert between char sets ----*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides a utility class to convert between different character
/// set encodings.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CHARSET_H
#define LLVM_SUPPORT_CHARSET_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/config.h"
#include "llvm/Support/ErrorOr.h"

#include <functional>
#include <string>
#include <system_error>

namespace llvm {

template <typename T> class SmallVectorImpl;

namespace details {
class CharSetConverterImplBase {
public:
  virtual ~CharSetConverterImplBase() = default;

  /// Converts a string.
  /// \param[in] Source source string
  /// \param[in,out] Result container for converted string
  /// \param[in] ShouldAutoFlush Append shift-back sequence after conversion
  /// for multi-byte encodings iff true.
  /// \return error code in case something went wrong
  ///
  /// The following error codes can occur, among others:
  ///   - std::errc::argument_list_too_long: The result requires more than
  ///     std::numeric_limits<size_t>::max() bytes.
  ///   - std::errc::illegal_byte_sequence: The input contains an invalid
  ///     multibyte sequence.
  ///   - std::errc::invalid_argument: The input contains an incomplete
  ///     multibyte sequence.
  ///
  /// In case of an error, the result string contains the successfully converted
  /// part of the input string.
  ///

  virtual std::error_code convert(StringRef Source,
                                  SmallVectorImpl<char> &Result,
                                  bool ShouldAutoFlush) const = 0;

  /// Restore the conversion to the original state.
  /// \return error code in case something went wrong
  ///
  /// If the original character set or the destination character set
  /// are multi-byte character sets, set the shift state to the initial
  /// state. Otherwise this is a no-op.
  virtual std::error_code flush() const = 0;

  virtual std::error_code flush(SmallVectorImpl<char> &Result) const = 0;
};
} // namespace details

// Names inspired by https://wg21.link/p1885.
namespace text_encoding {
enum class id {
  /// UTF-8 character set encoding.
  UTF8,

  /// IBM EBCDIC 1047 character set encoding.
  IBM1047
};
} // end namespace text_encoding

/// Utility class to convert between different character set encodings.
/// The class always supports converting between EBCDIC 1047 and Latin-1/UTF-8.
class CharSetConverter {
  // details::CharSetConverterImplBase *Converter;
  std::unique_ptr<details::CharSetConverterImplBase> Converter;

  CharSetConverter(std::unique_ptr<details::CharSetConverterImplBase> Converter)
      : Converter(std::move(Converter)) {}

public:
  /// Creates a CharSetConverter instance.
  /// \param[in] CSFrom name of the source character encoding
  /// \param[in] CSTo name of the target character encoding
  /// \return a CharSetConverter instance
  static CharSetConverter create(text_encoding::id CSFrom,
                                 text_encoding::id CSTo);

  /// Creates a CharSetConverter instance.
  /// Returns std::errc::invalid_argument in case the requested conversion is
  /// not supported.
  /// \param[in] CPFrom name of the source character encoding
  /// \param[in] CPTo name of the target character encoding
  /// \return a CharSetConverter instance or an error code
  static ErrorOr<CharSetConverter> create(StringRef CPFrom, StringRef CPTo);

  CharSetConverter(const CharSetConverter &) = delete;
  CharSetConverter &operator=(const CharSetConverter &) = delete;

  CharSetConverter(CharSetConverter &&Other) {
    Converter = std::move(Other.Converter);
  }

  CharSetConverter &operator=(CharSetConverter &&Other) {
    if (this != &Other)
      Converter = std::move(Other.Converter);
    return *this;
  }

  ~CharSetConverter() = default;

  /// Converts a string.
  /// \param[in] Source source string
  /// \param[in,out] Result container for converted string
  /// \param[in] ShouldAutoFlush Append shift-back sequence after conversion
  /// for multi-byte encodings.
  /// \return error code in case something went wrong
  std::error_code convert(StringRef Source, SmallVectorImpl<char> &Result,
                          bool ShouldAutoFlush = true) const {
    return Converter->convert(Source, Result, ShouldAutoFlush);
  }

  char convert(char SingleChar) const {
    SmallString<1> Result;
    Converter->convert(StringRef(&SingleChar, 1), Result, false);
    return Result[0];
  }

  /// Converts a string.
  /// \param[in] Source source string
  /// \param[in,out] Result container for converted string
  /// \param[in] ShouldAutoFlush Append shift-back sequence after conversion
  /// for multi-byte encodings iff true.
  /// \return error code in case something went wrong
  std::error_code convert(const std::string &Source,
                          SmallVectorImpl<char> &Result,
                          bool ShouldAutoFlush = true) const {
    return convert(StringRef(Source), Result, ShouldAutoFlush);
  }

  std::error_code flush() const { return Converter->flush(); }

  std::error_code flush(SmallVectorImpl<char> &Result) const {
    return Converter->flush(Result);
  }
};

} // namespace llvm

#endif
