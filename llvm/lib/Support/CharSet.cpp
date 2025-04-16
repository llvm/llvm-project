//===-- CharSet.cpp - Characters sets conversion class ------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides utility classes to convert between different character
/// set encodings.
///
//===----------------------------------------------------------------------===//

#include "llvm/Support/CharSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ConvertEBCDIC.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <limits>
#include <system_error>

#ifdef HAVE_ICU
#include <unicode/ucnv.h>
#elif defined(HAVE_ICONV)
#include <iconv.h>
#endif

using namespace llvm;

// Normalize the charset name with the charset alias matching algorithm proposed
// in https://www.unicode.org/reports/tr22/tr22-8.html#Charset_Alias_Matching.
static void normalizeCharSetName(StringRef CSName,
                                 SmallVectorImpl<char> &Normalized) {
  bool PrevDigit = false;
  for (auto Ch : CSName) {
    if (isAlnum(Ch)) {
      Ch = toLower(Ch);
      if (Ch != '0' || PrevDigit) {
        PrevDigit = isDigit(Ch);
        Normalized.push_back(Ch);
      }
    }
  }
}

// Maps the charset name to enum constant if possible.
static std::optional<text_encoding::id> getKnownCharSet(StringRef CSName) {
  SmallString<16> Normalized;
  normalizeCharSetName(CSName, Normalized);
  if (Normalized.equals("utf8"))
    return text_encoding::id::UTF8;
  if (Normalized.equals("ibm1047"))
    return text_encoding::id::IBM1047;
  return std::nullopt;
}

LLVM_ATTRIBUTE_UNUSED static void
HandleOverflow(size_t &Capacity, char *&Output, size_t &OutputLength,
               SmallVectorImpl<char> &Result) {
  // No space left in output buffer. Double the size of the underlying
  // memory in the SmallVectorImpl, adjust pointer and length and continue
  // the conversion.
  Capacity = (Capacity < std::numeric_limits<size_t>::max() / 2)
                 ? 2 * Capacity
                 : std::numeric_limits<size_t>::max();
  Result.resize(0);
  Result.resize_for_overwrite(Capacity);
  Output = static_cast<char *>(Result.data());
  OutputLength = Capacity;
}

namespace {
enum ConversionType {
  UTF8ToIBM1047,
  IBM1047ToUTF8,
};

// Support conversion between EBCDIC 1047 and UTF-8. This class uses
// built-in translation tables that allow for translation between the
// aforementioned character sets. The use of tables for conversion is only
// possible because EBCDIC 1047 is a single-byte, stateless encoding; other
// character sets are not supported.
class CharSetConverterTable : public details::CharSetConverterImplBase {
  const ConversionType ConvType;

public:
  CharSetConverterTable(ConversionType ConvType) : ConvType(ConvType) {}

  std::error_code convertString(StringRef Source,
                                SmallVectorImpl<char> &Result) override;

  void reset() override {}
};

std::error_code
CharSetConverterTable::convertString(StringRef Source,
                                     SmallVectorImpl<char> &Result) {
  if (ConvType == IBM1047ToUTF8) {
    ConverterEBCDIC::convertToUTF8(Source, Result);
    return std::error_code();
  } else if (ConvType == UTF8ToIBM1047) {
    return ConverterEBCDIC::convertToEBCDIC(Source, Result);
  }
  llvm_unreachable("Invalid ConvType!");
  return std::error_code();
}

#ifdef HAVE_ICU
struct UConverterDeleter {
  void operator()(UConverter *Converter) const {
    if (Converter)
      ucnv_close(Converter);
  }
};
using UConverterUniquePtr = std::unique_ptr<UConverter, UConverterDeleter>;

class CharSetConverterICU : public details::CharSetConverterImplBase {
  UConverterUniquePtr FromConvDesc;
  UConverterUniquePtr ToConvDesc;

public:
  CharSetConverterICU(UConverterUniquePtr FromConverter,
                      UConverterUniquePtr ToConverter)
      : FromConvDesc(std::move(FromConverter)),
        ToConvDesc(std::move(ToConverter)) {}

  std::error_code convertString(StringRef Source,
                                SmallVectorImpl<char> &Result) override;

  void reset() override;
};

std::error_code
CharSetConverterICU::convertString(StringRef Source,
                                   SmallVectorImpl<char> &Result) {
  // Setup the input in case it has no backing data.
  size_t InputLength = Source.size();
  const char *In = InputLength ? const_cast<char *>(Source.data()) : "";

  // Setup the output. We directly write into the SmallVector.
  size_t Capacity = Result.capacity();
  size_t OutputLength = Capacity;
  Result.resize_for_overwrite(Capacity);
  char *Output = static_cast<char *>(Result.data());
  UErrorCode EC = U_ZERO_ERROR;

  ucnv_setToUCallBack(&*FromConvDesc, UCNV_TO_U_CALLBACK_STOP, NULL, NULL, NULL,
                      &EC);
  ucnv_setFromUCallBack(&*ToConvDesc, UCNV_FROM_U_CALLBACK_STOP, NULL, NULL,
                        NULL, &EC);
  assert(U_SUCCESS(EC));

  do {
    EC = U_ZERO_ERROR;
    const char *Input = In;

    Output = InputLength ? static_cast<char *>(Result.data()) : nullptr;
    ucnv_convertEx(&*ToConvDesc, &*FromConvDesc, &Output, Result.end(), &Input,
                   In + InputLength, /*pivotStart=*/NULL,
                   /*pivotSource=*/NULL, /*pivotTarget=*/NULL,
                   /*pivotLimit=*/NULL, /*reset=*/true,
                   /*flush=*/true, &EC);
    if (U_FAILURE(EC)) {
      if (EC == U_BUFFER_OVERFLOW_ERROR &&
          Capacity < std::numeric_limits<size_t>::max()) {
        HandleOverflow(Capacity, Output, OutputLength, Result);
        continue;
      }
      // Some other error occured.
      Result.resize(Output - Result.data());
      return std::error_code(EILSEQ, std::generic_category());
    }
    break;
  } while (true);

  Result.resize(Output - Result.data());
  return std::error_code();
}

void CharSetConverterICU::reset() {
  ucnv_reset(&*FromConvDesc);
  ucnv_reset(&*ToConvDesc);
}

#elif defined(HAVE_ICONV)
class CharSetConverterIconv : public details::CharSetConverterImplBase {
  class UniqueIconvT {
    iconv_t ConvDesc;

  public:
    operator iconv_t() const { return ConvDesc; }
    UniqueIconvT(iconv_t CD) : ConvDesc(CD) {}
    ~UniqueIconvT() {
      if (ConvDesc != (iconv_t)-1) {
        iconv_close(ConvDesc);
        ConvDesc = (iconv_t)-1;
      }
    }
    UniqueIconvT(UniqueIconvT &&Other) : ConvDesc(Other.ConvDesc) {
      Other.ConvDesc = (iconv_t)-1;
    }
    UniqueIconvT &operator=(UniqueIconvT &&Other) {
      if (&Other != this) {
        ConvDesc = Other.ConvDesc;
        Other.ConvDesc = (iconv_t)-1;
      }
      return *this;
    }
  };
  UniqueIconvT ConvDesc;

public:
  CharSetConverterIconv(UniqueIconvT ConvDesc)
      : ConvDesc(std::move(ConvDesc)) {}

  std::error_code convertString(StringRef Source,
                                SmallVectorImpl<char> &Result) override;

  void reset() override;
};

std::error_code
CharSetConverterIconv::convertString(StringRef Source,
                                     SmallVectorImpl<char> &Result) {
  // Setup the output. We directly write into the SmallVector.
  size_t Capacity = Result.capacity();
  char *Output = static_cast<char *>(Result.data());
  size_t OutputLength = Capacity;
  Result.resize_for_overwrite(Capacity);

  size_t Ret;
  // Handle errors returned from iconv().
  auto HandleError = [&Capacity, &Output, &OutputLength, &Result,
                      this](size_t Ret) {
    if (Ret == static_cast<size_t>(-1)) {
      // An error occured. Check if we can gracefully handle it.
      if (errno == E2BIG && Capacity < std::numeric_limits<size_t>::max()) {
        HandleOverflow(Capacity, Output, OutputLength, Result);
        // Reset converter
        iconv(ConvDesc, nullptr, nullptr, nullptr, nullptr);
        return std::error_code();
      } else {
        // Some other error occured.
        Result.resize(Output - Result.data());
        return std::error_code(errno, std::generic_category());
      }
    } else {
      // A positive return value indicates that some characters were converted
      // in a nonreversible way, that is, replaced with a SUB symbol. Returning
      // an error in this case makes sure that both conversion routines behave
      // in the same way.
      return std::make_error_code(std::errc::illegal_byte_sequence);
    }
  };

  do {
    // Setup the input. Use nullptr to reset iconv state if input length is
    // zero.
    size_t InputLength = Source.size();
    char *Input = InputLength ? const_cast<char *>(Source.data()) : nullptr;
    Ret = iconv(ConvDesc, &Input, &InputLength, &Output, &OutputLength);
    if (Ret != 0) {
      if (auto EC = HandleError(Ret))
        return EC;
      continue;
    }
    // Flush the converter
    Ret = iconv(ConvDesc, nullptr, nullptr, &Output, &OutputLength);
    if (Ret != 0) {
      if (auto EC = HandleError(Ret))
        return EC;
      continue;
    }
    break;
  } while (true);

  // Re-adjust size to actual size.
  Result.resize(Output - Result.data());
  return std::error_code();
}

void CharSetConverterIconv::reset() {
  iconv(ConvDesc, nullptr, nullptr, nullptr, nullptr);
}

#endif // HAVE_ICONV
} // namespace

ErrorOr<CharSetConverter> CharSetConverter::create(text_encoding::id CPFrom,
                                                   text_encoding::id CPTo) {

  assert(CPFrom != CPTo && "Text encodings should be distinct");

  ConversionType Conversion;
  if (CPFrom == text_encoding::id::UTF8 && CPTo == text_encoding::id::IBM1047)
    Conversion = UTF8ToIBM1047;
  else if (CPFrom == text_encoding::id::IBM1047 &&
           CPTo == text_encoding::id::UTF8)
    Conversion = IBM1047ToUTF8;
  else
    return std::error_code(errno, std::generic_category());

  std::unique_ptr<details::CharSetConverterImplBase> Converter =
      std::make_unique<CharSetConverterTable>(Conversion);
  return CharSetConverter(std::move(Converter));
}

ErrorOr<CharSetConverter> CharSetConverter::create(StringRef CSFrom,
                                                   StringRef CSTo) {
  std::optional<text_encoding::id> From = getKnownCharSet(CSFrom);
  std::optional<text_encoding::id> To = getKnownCharSet(CSTo);
  if (From && To) {
    ErrorOr<CharSetConverter> Converter = create(*From, *To);
    if (Converter)
      return Converter;
  }
#ifdef HAVE_ICU
  UErrorCode EC = U_ZERO_ERROR;
  UConverterUniquePtr FromConvDesc(ucnv_open(CSFrom.str().c_str(), &EC));
  if (U_FAILURE(EC)) {
    return std::error_code(errno, std::generic_category());
  }
  UConverterUniquePtr ToConvDesc(ucnv_open(CSTo.str().c_str(), &EC));
  if (U_FAILURE(EC)) {
    return std::error_code(errno, std::generic_category());
  }
  std::unique_ptr<details::CharSetConverterImplBase> Converter =
      std::make_unique<CharSetConverterICU>(std::move(FromConvDesc),
                                            std::move(ToConvDesc));
  return CharSetConverter(std::move(Converter));
#elif defined(HAVE_ICONV)
  iconv_t ConvDesc = iconv_open(CSTo.str().c_str(), CSFrom.str().c_str());
  if (ConvDesc == (iconv_t)-1)
    return std::error_code(errno, std::generic_category());
  std::unique_ptr<details::CharSetConverterImplBase> Converter =
      std::make_unique<CharSetConverterIconv>(ConvDesc);
  return CharSetConverter(std::move(Converter));
#else
  return std::make_error_code(std::errc::invalid_argument);
#endif
}
