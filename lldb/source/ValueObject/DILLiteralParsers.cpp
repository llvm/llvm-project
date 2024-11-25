//===-- DILLiteralParsers.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This implements the recursive descent parser for the Data Inspection
// Language (DIL), and its helper functions, which will eventually underlie the
// 'frame variable' command. The language that this parser recognizes is
// described in lldb/docs/dil-expr-lang.ebnf
//
//===----------------------------------------------------------------------===//

#include "lldb/ValueObject/DILLiteralParsers.h"

#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Unicode.h"

#include <algorithm>
#include <iterator>
#include <string>

namespace lldb_private {

namespace dil {

enum class diag {
  // errs
  err_bad_character_encoding,
  err_bad_string_encoding,
  err_character_too_large,
  err_delimited_escape_empty,
  err_delimited_escape_invalid,
  err_delimited_escape_missing_brace,
  err_digit_separator_not_between_digits,
  err_escape_too_large,
  err_expected,
  err_exponent_has_no_digits,
  err_hex_constant_requires,
  err_hex_escape_no_digits,
  err_hex_literal_requires,
  err_invalid_digit,
  err_invalid_suffix_constant,
  err_invalid_ucn_name,
  err_lexing_char,
  err_lexing_numeric,
  err_lexing_string,
  err_multichar_character_literal,
  err_ucn_control_character,
  err_ucn_escape_basic_scs,
  err_ucn_escape_incomplete,
  err_ucn_escape_invalid,
  err_unevaluated_string_invalid_escape_sequence,
  err_unsupported_string_concat,
  // exts
  ext_binary_literal,
  ext_delimited_escape_sequence,
  ext_hex_literal_invalid,
  ext_nonstandard_escape,
  ext_string_too_long,
  ext_unknown_escape,
  // notes
  note_invalid_ucn_name_candidate,
  note_invalid_ucn_name_loose_matching,
  note_invalid_name_candidate,
  // warns
  warn_bad_character_encoding,
  warn_bad_string_encoding,
  warn_c23_compat_literal_ucn_control_character,
  warn_c23_compat_literal_ucn_escape_basic_scs,
  warn_char_constant_too_large,
  warn_four_char_character_literal,
  warn_multichar_character_literal,
  warn_ucn_not_valid_in_c89_literal,
  warn_unevaluated_string_prefix,
};

static void Diags_Report(uint32_t loc, dil::diag diag_id, std::string diag_msg)
{
}


static unsigned getCharWidth(dil::TokenKind kind,
                             const clang::TargetInfo &Target) {
  switch (kind) {
  default: llvm_unreachable("Unknown token type!");
  case dil::TokenKind::char_constant:
  case dil::TokenKind::string_literal:
  case dil::TokenKind::utf8_char_constant:
  case dil::TokenKind::utf8_string_literal:
    return Target.getCharWidth();
  case dil::TokenKind::wide_char_constant:
  case dil::TokenKind::wide_string_literal:
    return Target.getWCharWidth();
  }
}

NumericLiteralParser::NumericLiteralParser(llvm::StringRef TokSpelling,
                                           unsigned TokLoc,
                                           bool AllowHalfType,
                                           //bool AllowFixedPoint,
                                           DILLexer &lexer,
                                           bool AllowMicrosoftExt) :
    ThisTokBegin(TokSpelling.begin()), ThisTokEnd(TokSpelling.end()) {
  s = DigitsBegin = ThisTokBegin;
  saw_exponent = false;
  saw_period = false;
  isLong = false;
  isUnsigned = false;
  isLongLong = false;
  isSizeT = false;
  isHalf = false;
  isFloat = false;
  isImaginary = false;
  isFloat16 = false;
  isFloat128 = false;
  MicrosoftInteger = 0;
  isFract = false;
  isAccum = false;
  hadError = false;
  isBitInt = false;

  // This routine assumes that the range begin/end matches the regex for integer
  // and FP constants (specifically, the 'pp-number' regex), and assumes that
  // the byte at "*end" is both valid and not part of the regex.  Because of
  // this, it doesn't have to check for 'overscan' in various places.
  // Note: For HLSL, the end token is allowed to be '.' which would be in the
  // 'pp-number' regex. This is required to support vector swizzles on numeric
  // constants (i.e. 1.xx or 1.5f.rrr).
  if (isPreprocessingNumberBody(*ThisTokEnd) &&
      !(*ThisTokEnd == '.')) {
    Diags_Report(TokLoc, diag::err_lexing_numeric, "");
    hadError = true;
    return;
  }

  if (*s == '0') { // parse radix
    ParseNumberStartingWithZero(TokLoc, /*AllowHexFloats=*/true);
    if (hadError)
      return;
  } else { // the first digit is non-zero
    radix = 10;
    s = SkipDigits(s);
    if (s == ThisTokEnd) {
      // Done.
    } else {
      ParseDecimalOrOctalCommon(TokLoc);
      if (hadError)
        return;
    }
  }

  SuffixBegin = s;
  CheckSeparator(TokLoc, s, CSK_AfterDigits);

  // Parse the suffix.  At this point we can classify whether we have an FP or
  // integer constant.
  bool isFixedPointConstant = isFixedPointLiteral();
  bool isFPConstant = isFloatingLiteral();
  bool HasSize = false;
  bool DoubleUnderscore = false;

  // Loop over all of the characters of the suffix.  If we see something bad,
  // we break out of the loop.
  for (; s != ThisTokEnd; ++s) {
    switch (*s) {
    case 'R':
    case 'r':
      //if (!AllowFixedPoint)
      //  break;
      if (isFract || isAccum) break;
      if (!(saw_period || saw_exponent)) break;
      isFract = true;
      continue;
    case 'K':
    case 'k':
      //if (!AllowFixedPoint)
      //  break;
      if (isFract || isAccum) break;
      if (!(saw_period || saw_exponent)) break;
      isAccum = true;
      continue;
    case 'h':      // FP Suffix for "half".
    case 'H':
      // OpenCL Extension v1.2 s9.5 - h or H suffix for half type.
      if (!(AllowHalfType))// || AllowFixedPoint))
        break;
      if (isIntegerLiteral()) break;  // Error for integer constant.
      if (HasSize)
        break;
      HasSize = true;
      isHalf = true;
      continue;  // Success.
    case 'f':      // FP Suffix for "float"
    case 'F':
      if (!isFPConstant) break;  // Error for integer constant.
      if (HasSize)
        break;
      HasSize = true;

      // CUDA host and device may have different _Float16 support, therefore
      // allows f16 literals to avoid false alarm.
      // When we compile for OpenMP target offloading on NVPTX, f16 suffix
      // should also be supported.
      // ToDo: more precise check for CUDA.
      // TODO: AMDGPU might also support it in the future.
      if ((lexer.getTargetInfo().hasFloat16Type() ||
               lexer.getTargetInfo().getTriple().isNVPTX()) &&
          s + 2 < ThisTokEnd && s[1] == '1' && s[2] == '6') {
        s += 2; // success, eat up 2 characters.
        isFloat16 = true;
        continue;
      }

      isFloat = true;
      continue;  // Success.
    case 'q':    // FP Suffix for "__float128"
    case 'Q':
      if (!isFPConstant) break;  // Error for integer constant.
      if (HasSize)
        break;
      HasSize = true;
      isFloat128 = true;
      continue;  // Success.
    case 'u':
    case 'U':
      if (isFPConstant) break;  // Error for floating constant.
      if (isUnsigned) break;    // Cannot be repeated.
      isUnsigned = true;
      continue;  // Success.
    case 'l':
    case 'L':
      if (HasSize)
        break;
      HasSize = true;

      // Check for long long.  The L's need to be adjacent and the same case.
      if (s[1] == s[0]) {
        assert(s + 1 < ThisTokEnd && "didn't maximally munch?");
        if (isFPConstant) break;        // long long invalid for floats.
        isLongLong = true;
        ++s;  // Eat both of them.
      } else {
        isLong = true;
      }
      continue; // Success.
    case 'z':
    case 'Z':
      if (isFPConstant)
        break; // Invalid for floats.
      if (HasSize)
        break;
      HasSize = true;
      isSizeT = true;
      continue;
    case 'i':
    case 'I':
      if (AllowMicrosoftExt && !isFPConstant) {
        // Allow i8, i16, i32, and i64. First, look ahead and check if
        // suffixes are Microsoft integers and not the imaginary unit.
        uint8_t Bits = 0;
        size_t ToSkip = 0;
        switch (s[1]) {
        case '8': // i8 suffix
          Bits = 8;
          ToSkip = 2;
          break;
        case '1':
          if (s[2] == '6') { // i16 suffix
            Bits = 16;
            ToSkip = 3;
          }
          break;
        case '3':
          if (s[2] == '2') { // i32 suffix
            Bits = 32;
            ToSkip = 3;
          }
          break;
        case '6':
          if (s[2] == '4') { // i64 suffix
            Bits = 64;
            ToSkip = 3;
          }
          break;
        default:
          break;
        }
        if (Bits) {
          if (HasSize)
            break;
          HasSize = true;
          MicrosoftInteger = Bits;
          s += ToSkip;
          assert(s <= ThisTokEnd && "didn't maximally munch?");
          break;
        }
      }
      [[fallthrough]];
    case 'j':
    case 'J':
      if (isImaginary) break;   // Cannot be repeated.
      isImaginary = true;
      continue;  // Success.
    case '_':
      if (isFPConstant)
        break; // Invalid for floats
      if (HasSize)
        break;
      // There is currently no way to reach this with DoubleUnderscore set.
      // If new double underscope literals are added handle it here as above.
      assert(!DoubleUnderscore && "unhandled double underscore case");
      if (s + 2 < ThisTokEnd &&
          s[1] == '_') { // s + 2 < ThisTokEnd to ensure some character exists
                         // after __
        DoubleUnderscore = true;
        s += 2; // Skip both '_'
        if (s + 1 < ThisTokEnd &&
            (*s == 'u' || *s == 'U')) { // Ensure some character after 'u'/'U'
          isUnsigned = true;
          ++s;
        }
        if (s + 1 < ThisTokEnd &&
            ((*s == 'w' && *(++s) == 'b') || (*s == 'W' && *(++s) == 'B'))) {
          isBitInt = true;
          HasSize = true;
          continue;
        }
      }
      break;
    case 'w':
    case 'W':
      if (isFPConstant)
        break; // Invalid for floats.
      if (HasSize)
        break; // Invalid if we already have a size for the literal.

      // wb and WB are allowed, but a mixture of cases like Wb or wB is not. We
      // explicitly do not support the suffix in C++ as an extension because a
      // library-based UDL that resolves to a library type may be more
      // appropriate there. The same rules apply for __wb/__WB.
      if ((DoubleUnderscore && s + 1 < ThisTokEnd &&
           ((s[0] == 'w' && s[1] == 'b') || (s[0] == 'W' && s[1] == 'B')))) {
        isBitInt = true;
        HasSize = true;
        ++s; // Skip both characters (2nd char skipped on continue).
        continue; // Success.
      }
    }
    // If we reached here, there was an error or a ud-suffix.
    break;
  }

  if (s != ThisTokEnd) {
    // Report an error if there are any.
    std::string err_msg =
        llvm::StringRef(SuffixBegin, ThisTokEnd - SuffixBegin).str();
    err_msg += (isFixedPointConstant ? "2"
                : (isFPConstant ? "1" : "0"));
    //Diags_Report(Lexer::AdvanceToTokenCharacter(
    //                 TokLoc, SuffixBegin - ThisTokBegin, SM, LangOpts),
    Diags_Report(SuffixBegin - ThisTokBegin - TokLoc,
                 diag::err_invalid_suffix_constant, err_msg);
        hadError = true;
  }

  if (!hadError && saw_fixed_point_suffix) {
    assert(isFract || isAccum);
  }
}


static bool alwaysFitsInto64Bits(unsigned Radix, unsigned NumDigits) {
  switch (Radix) {
  case 2:
    return NumDigits <= 64;
  case 8:
    return NumDigits <= 64 / 3; // Digits are groups of 3 bits.
  case 10:
    return NumDigits <= 19; // floor(log10(2^64))
  case 16:
    return NumDigits <= 64 / 4; // Digits are groups of 4 bits.
  default:
    llvm_unreachable("impossible Radix");
  }
}

bool NumericLiteralParser::GetIntegerValue(llvm::APInt &Val) {
  // Fast path: Compute a conservative bound on the maximum number of
  // bits per digit in this radix. If we can't possibly overflow a
  // uint64 based on that bound then do the simple conversion to
  // integer. This avoids the expensive overflow checking below, and
  // handles the common cases that matter (small decimal integers and
  // hex/octal values which don't overflow).
  const unsigned NumDigits = SuffixBegin - DigitsBegin;
  if (alwaysFitsInto64Bits(radix, NumDigits)) {
    uint64_t N = 0;
    for (const char *Ptr = DigitsBegin; Ptr != SuffixBegin; ++Ptr)
      if (!isDigitSeparator(*Ptr))
        N = N * radix + llvm::hexDigitValue(*Ptr);

    // This will truncate the value to Val's input width. Simply check
    // for overflow by comparing.
    Val = N;
    return Val.getZExtValue() != N;
  }

  Val = 0;
  const char *Ptr = DigitsBegin;

  llvm::APInt RadixVal(Val.getBitWidth(), radix);
  llvm::APInt CharVal(Val.getBitWidth(), 0);
  llvm::APInt OldVal = Val;

  bool OverflowOccurred = false;
  while (Ptr < SuffixBegin) {
    if (isDigitSeparator(*Ptr)) {
      ++Ptr;
      continue;
    }

    unsigned C = llvm::hexDigitValue(*Ptr++);

    // If this letter is out of bound for this radix, reject it.
    assert(C < radix && "NumericLiteralParser ctor should have rejected this");

    CharVal = C;

    // Add the digit to the value in the appropriate radix.  If adding in digits
    // made the value smaller, then this overflowed.
    OldVal = Val;

    // Multiply by radix, did overflow occur on the multiply?
    Val *= RadixVal;
    OverflowOccurred |= Val.udiv(RadixVal) != OldVal;

    // Add value, did overflow occur on the value?
    //   (a + b) ult b  <=> overflow
    Val += CharVal;
    OverflowOccurred |= Val.ult(CharVal);
  }
  return OverflowOccurred;
}

llvm::APFloat::opStatus NumericLiteralParser::GetFloatValue(
    llvm::APFloat &Result,
    llvm::RoundingMode RM) {
  using llvm::APFloat;

  unsigned n = std::min(SuffixBegin - ThisTokBegin, ThisTokEnd - ThisTokBegin);

  llvm::SmallString<16> Buffer;
  llvm::StringRef Str(ThisTokBegin, n);
  if (Str.contains('\'')) {
    Buffer.reserve(n);
    std::remove_copy_if(Str.begin(), Str.end(), std::back_inserter(Buffer),
                        &isDigitSeparator);
    Str = Buffer;
  }

  auto StatusOrErr = Result.convertFromString(Str, RM);
  assert(StatusOrErr && "Invalid floating point representation");
  return !errorToBool(StatusOrErr.takeError()) ? *StatusOrErr
                                               : APFloat::opInvalidOp;
}

void NumericLiteralParser::ParseNumberStartingWithZero(unsigned TokLoc,
                                                       bool AllowHexFloats) {
  assert(s[0] == '0' && "Invalid method call");
  s++;

  int c1 = s[0];

  // Handle a hex number like 0x1234.
  if ((c1 == 'x' || c1 == 'X') && (isHexDigit(s[1]) || s[1] == '.')) {
    s++;
    assert(s < ThisTokEnd && "didn't maximally munch?");
    radix = 16;
    DigitsBegin = s;
    s = SkipHexDigits(s);
    bool HasSignificandDigits = containsDigits(DigitsBegin, s);
    if (s == ThisTokEnd) {
      // Done.
    } else if (*s == '.') {
      s++;
      saw_period = true;
      const char *floatDigitsBegin = s;
      s = SkipHexDigits(s);
      if (containsDigits(floatDigitsBegin, s))
        HasSignificandDigits = true;
      if (HasSignificandDigits)
        CheckSeparator(TokLoc, floatDigitsBegin, CSK_BeforeDigits);
    }

    if (!HasSignificandDigits) {
      //Diags_Report(Lexer::AdvanceToTokenCharacter(TokLoc, s - ThisTokBegin, SM,
      //                                            LangOpts),
      Diags_Report(s - ThisTokBegin - TokLoc,
                   diag::err_hex_constant_requires, "1");
      hadError = true;
      return;
    }

    // A binary exponent can appear with or with a '.'. If dotted, the
    // binary exponent is required.
    if (*s == 'p' || *s == 'P') {
      CheckSeparator(TokLoc, s, CSK_AfterDigits);
      const char *Exponent = s;
      s++;
      saw_exponent = true;
      if (s != ThisTokEnd && (*s == '+' || *s == '-'))  s++; // sign
      const char *first_non_digit = SkipDigits(s);
      if (!containsDigits(s, first_non_digit)) {
        if (!hadError) {
          //Diags_Report(Lexer::AdvanceToTokenCharacter(
          //                 TokLoc, Exponent - ThisTokBegin, SM, LangOpts),
          Diags_Report(Exponent - ThisTokBegin - TokLoc,
                       diag::err_exponent_has_no_digits, "");
          hadError = true;
        }
        return;
      }
      CheckSeparator(TokLoc, s, CSK_BeforeDigits);
      s = first_non_digit;

      if (!AllowHexFloats)
        Diags_Report(TokLoc, diag::ext_hex_literal_invalid, "");
    } else if (saw_period) {
      //Diags_Report(Lexer::AdvanceToTokenCharacter(TokLoc, s - ThisTokBegin, SM,
      //                                            LangOpts),
      Diags_Report(s - ThisTokBegin - TokLoc,
                   diag::err_hex_constant_requires, "0");
      hadError = true;
    }
    return;
  }

  // Handle simple binary numbers 0b01010
  if ((c1 == 'b' || c1 == 'B') && (s[1] == '0' || s[1] == '1')) {
    // 0b101010 is a C++14 and C23 extension.
    dil::diag DiagId;
    DiagId = diag::ext_binary_literal;
    Diags_Report(TokLoc, DiagId, "");
    ++s;
    assert(s < ThisTokEnd && "didn't maximally munch?");
    radix = 2;
    DigitsBegin = s;
    s = SkipBinaryDigits(s);
    if (s == ThisTokEnd) {
      // Done.
    } else if (isHexDigit(*s)) {
      std::string err_msg = llvm::StringRef(s, 1).str() +  "2";
      //Diags_Report(Lexer::AdvanceToTokenCharacter(TokLoc, s - ThisTokBegin, SM,
      //                                            LangOpts),
      Diags_Report(s - ThisTokBegin - TokLoc,
                   diag::err_invalid_digit, err_msg);
      hadError = true;
    }
    return;
  }

  // For now, the radix is set to 8. If we discover that we have a
  // floating point constant, the radix will change to 10. Octal floating
  // point constants are not permitted (only decimal and hexadecimal).
  radix = 8;
  const char *PossibleNewDigitStart = s;
  s = SkipOctalDigits(s);
  // When the value is 0 followed by a suffix (like 0wb), we want to leave 0
  // as the start of the digits. So if skipping octal digits does not skip
  // anything, we leave the digit start where it was.
  if (s != PossibleNewDigitStart)
    DigitsBegin = PossibleNewDigitStart;

  if (s == ThisTokEnd)
    return; // Done, simple octal number like 01234

  // If we have some other non-octal digit that *is* a decimal digit, see if
  // this is part of a floating point number like 094.123 or 09e1.
  if (isDigit(*s)) {
    const char *EndDecimal = SkipDigits(s);
    if (EndDecimal[0] == '.' || EndDecimal[0] == 'e' || EndDecimal[0] == 'E') {
      s = EndDecimal;
      radix = 10;
    }
  }

  ParseDecimalOrOctalCommon(TokLoc);
}

void NumericLiteralParser::ParseDecimalOrOctalCommon(unsigned TokLoc) {
  assert((radix == 8 || radix == 10) && "Unexpected radix");

  // If we have a hex digit other than 'e' (which denotes a FP exponent) then
  // the code is using an incorrect base.
  if (isHexDigit(*s) && *s != 'e' && *s != 'E') {
    std::string err_msg = llvm::StringRef(s, 1).str();
    err_msg += (radix == 8 ? "1" : "0");
    Diags_Report(
        //Lexer::AdvanceToTokenCharacter(TokLoc, s - ThisTokBegin, SM, LangOpts),
        s - ThisTokBegin - TokLoc,
        diag::err_invalid_digit, err_msg);
    hadError = true;
    return;
  }

  if (*s == '.') {
    CheckSeparator(TokLoc, s, CSK_AfterDigits);
    s++;
    radix = 10;
    saw_period = true;
    CheckSeparator(TokLoc, s, CSK_BeforeDigits);
    s = SkipDigits(s); // Skip suffix.
  }
  if (*s == 'e' || *s == 'E') { // exponent
    CheckSeparator(TokLoc, s, CSK_AfterDigits);
    const char *Exponent = s;
    s++;
    radix = 10;
    saw_exponent = true;
    if (s != ThisTokEnd && (*s == '+' || *s == '-'))  s++; // sign
    const char *first_non_digit = SkipDigits(s);
    if (containsDigits(s, first_non_digit)) {
      CheckSeparator(TokLoc, s, CSK_BeforeDigits);
      s = first_non_digit;
    } else {
      if (!hadError) {
        //Diags_Report(Lexer::AdvanceToTokenCharacter(
        //                 TokLoc, Exponent - ThisTokBegin, SM, LangOpts),
        Diags_Report(Exponent - ThisTokBegin - TokLoc,
                     diag::err_exponent_has_no_digits, "");
        hadError = true;
      }
      return;
    }
  }
}

void NumericLiteralParser::CheckSeparator(unsigned TokLoc, const char *Pos,
                                          CheckSeparatorKind IsAfterDigits) {
  if (IsAfterDigits == CSK_AfterDigits) {
    if (Pos == ThisTokBegin)
      return;
    --Pos;
  } else if (Pos == ThisTokEnd)
    return;

  if (isDigitSeparator(*Pos)) {
    //Diags_Report(Lexer::AdvanceToTokenCharacter(TokLoc, Pos - ThisTokBegin, SM,
    //                                            LangOpts),
    Diags_Report(Pos - ThisTokBegin - TokLoc,
                 diag::err_digit_separator_not_between_digits,
                 (IsAfterDigits ? "true" : "false"));
    hadError = true;
  }
}


static void DiagnoseInvalidUnicodeCharacterName(uint32_t Loc,
                                                const char *TokBegin,
                                                const char *TokRangeBegin,
                                                const char *TokRangeEnd,
                                                llvm::StringRef Name) {

  Diags_Report(Loc,
               diag::err_invalid_ucn_name,
               Name.str());

  namespace u = llvm::sys::unicode;

  std::optional<u::LooseMatchingResult> Res =
      u::nameToCodepointLooseMatching(Name);
  if (Res) {
    Diags_Report(Loc,
                 diag::note_invalid_ucn_name_loose_matching, "");
    return;
  }

  unsigned Distance = 0;
  llvm::SmallVector<u::MatchForCodepointName> Matches =
      u::nearestMatchesForCodepointName(Name, 5);
  assert(!Matches.empty() && "No unicode characters found");

  for (const auto &Match : Matches) {
    if (Distance == 0)
      Distance = Match.Distance;
    if (std::max(Distance, Match.Distance) -
            std::min(Distance, Match.Distance) >
        3)
      break;
    Distance = Match.Distance;

    std::string Str;
    llvm::UTF32 V = Match.Value;
    bool Converted =
        llvm::convertUTF32ToUTF8String(llvm::ArrayRef<llvm::UTF32>(&V, 1), Str);
    (void)Converted;
    assert(Converted && "Found a match wich is not a unicode character");

    std::string err_msg = Match.Name + llvm::utohexstr(Match.Value);
    Diags_Report(Loc,
                 diag::note_invalid_ucn_name_candidate, err_msg);
  }
}

static bool ProcessNumericUCNEscape(const char *ThisTokBegin,
                                    const char *&ThisTokBuf,
                                    const char *ThisTokEnd, uint32_t &UcnVal,
                                    unsigned short &UcnLen, bool &Delimited,
                                    uint32_t Loc,
                                    bool in_char_string_literal = false) {
  const char *UcnBegin = ThisTokBuf;
  bool HasError = false;
  bool EndDelimiterFound = false;

  // Skip the '\u' char's.
  ThisTokBuf += 2;
  Delimited = false;
  if (UcnBegin[1] == 'u' && in_char_string_literal &&
      ThisTokBuf != ThisTokEnd && *ThisTokBuf == '{') {
    Delimited = true;
    ThisTokBuf++;
  } else if (ThisTokBuf == ThisTokEnd || !isHexDigit(*ThisTokBuf)) {
    Diags_Report(Loc,
                 diag::err_hex_escape_no_digits,
                 llvm::StringRef(&ThisTokBuf[-1], 1).str());
    return false;
  }
  UcnLen = (ThisTokBuf[-1] == 'u' ? 4 : 8);

  bool Overflow = false;
  unsigned short Count = 0;
  for (; ThisTokBuf != ThisTokEnd && (Delimited || Count != UcnLen);
       ++ThisTokBuf) {
    if (Delimited && *ThisTokBuf == '}') {
      ++ThisTokBuf;
      EndDelimiterFound = true;
      break;
    }
    int CharVal = llvm::hexDigitValue(*ThisTokBuf);
    if (CharVal == -1) {
      HasError = true;
      if (!Delimited)
        break;
      Diags_Report(Loc,
                   diag::err_delimited_escape_invalid,
                   llvm::StringRef(ThisTokBuf, 1).str());
      Count++;
      continue;
    }
    if (UcnVal & 0xF0000000) {
      Overflow = true;
      continue;
    }
    UcnVal <<= 4;
    UcnVal |= CharVal;
    Count++;
  }

  if (Overflow) {
    Diags_Report(Loc,
                 diag::err_escape_too_large, "");
    return false;
  }

  if (Delimited && !EndDelimiterFound) {
    Diags_Report(Loc,
                 diag::err_expected, "]");
    return false;
  }

  // If we didn't consume the proper number of digits, there is a problem.
  if (Count == 0 || (!Delimited && Count != UcnLen)) {
    Diags_Report(Loc,
                 Delimited ? diag::err_delimited_escape_empty
                 : diag::err_ucn_escape_incomplete, "");
    return false;
  }
  return !HasError;
}

static bool ProcessNamedUCNEscape(const char *ThisTokBegin,
                                  const char *&ThisTokBuf,
                                  const char *ThisTokEnd, uint32_t &UcnVal,
                                  unsigned short &UcnLen, uint32_t Loc) {
  const char *UcnBegin = ThisTokBuf;
  assert(UcnBegin[0] == '\\' && UcnBegin[1] == 'N');
  ThisTokBuf += 2;
  if (ThisTokBuf == ThisTokEnd || *ThisTokBuf != '{') {
    Diags_Report(Loc,
                 diag::err_delimited_escape_missing_brace,
                 llvm::StringRef(&ThisTokBuf[-1], 1).str());
    return false;
  }
  ThisTokBuf++;
  const char *ClosingBrace = std::find_if(ThisTokBuf, ThisTokEnd, [](char C) {
    return C == '}' || isVerticalWhitespace(C);
  });
  bool Incomplete = ClosingBrace == ThisTokEnd;
  bool Empty = ClosingBrace == ThisTokBuf;
  if (Incomplete || Empty) {
    Diags_Report(Loc,
                 Incomplete ? diag::err_ucn_escape_incomplete
                 : diag::err_delimited_escape_empty,
                 llvm::StringRef(&UcnBegin[1], 1).str());
    ThisTokBuf = ClosingBrace == ThisTokEnd ? ClosingBrace : ClosingBrace + 1;
    return false;
  }
  llvm::StringRef Name(ThisTokBuf, ClosingBrace - ThisTokBuf);
  ThisTokBuf = ClosingBrace + 1;
  std::optional<char32_t> Res = llvm::sys::unicode::nameToCodepointStrict(Name);
  if (!Res) {
    DiagnoseInvalidUnicodeCharacterName(Loc, ThisTokBegin,
                                        &UcnBegin[3], ClosingBrace, Name);
    return false;
  }
  UcnVal = *Res;
  UcnLen = UcnVal > 0xFFFF ? 8 : 4;
  return true;
}

static void EncodeUCNEscape(const char *ThisTokBegin, const char *&ThisTokBuf,
                            const char *ThisTokEnd,
                            char *&ResultBuf, bool &HadError,
                            uint32_t Loc, unsigned CharByteWidth) {
  // CAROLINE!! FILL THIS IN!!
}


/// ProcessUCNEscape - Read the Universal Character Name, check constraints and
/// return the UTF32.
static bool ProcessUCNEscape(const char *ThisTokBegin, const char *&ThisTokBuf,
                             const char *ThisTokEnd, uint32_t &UcnVal,
                             unsigned short &UcnLen, uint32_t Loc,
                             bool in_char_string_literal = false) {

  bool HasError;
  //const char *UcnBegin = ThisTokBuf;
  bool IsDelimitedEscapeSequence = false;
  bool IsNamedEscapeSequence = false;
  if (ThisTokBuf[1] == 'N') {
    IsNamedEscapeSequence = true;
    HasError = !ProcessNamedUCNEscape(ThisTokBegin, ThisTokBuf, ThisTokEnd,
                                      UcnVal, UcnLen, Loc);
  } else {
    HasError =
        !ProcessNumericUCNEscape(ThisTokBegin, ThisTokBuf, ThisTokEnd, UcnVal,
                                 UcnLen, IsDelimitedEscapeSequence, Loc,
                                 in_char_string_literal);
  }
  if (HasError)
    return false;

  // Check UCN constraints (C99 6.4.3p2) [C++11 lex.charset p2]
  if ((0xD800 <= UcnVal && UcnVal <= 0xDFFF) || // surrogate codepoints
      UcnVal > 0x10FFFF) {                      // maximum legal UTF32 value
    Diags_Report(Loc,
                 diag::err_ucn_escape_invalid, "");
    return false;
  }

  // C23 and C++11 allow UCNs that refer to control characters
  // and basic source characters inside character and string literals
  if (UcnVal < 0xa0 &&
      // $, @, ` are allowed in all language modes
      (UcnVal != 0x24 && UcnVal != 0x40 && UcnVal != 0x60)) {
    bool IsError = !in_char_string_literal;
    char BasicSCSChar = UcnVal;
    if (UcnVal >= 0x20 && UcnVal < 0x7f) {
      std::string err_msg = llvm::StringRef(&BasicSCSChar, 1).str();
      Diags_Report(Loc,
                   IsError ? diag::err_ucn_escape_basic_scs
                   : diag::warn_c23_compat_literal_ucn_escape_basic_scs,
                   err_msg);
    } else {
      Diags_Report(Loc,
                   IsError ? diag::err_ucn_control_character
                   : diag::warn_c23_compat_literal_ucn_control_character, "");
    }
    if (IsError)
      return false;
  }

  Diags_Report(Loc,
               diag::warn_ucn_not_valid_in_c89_literal, "");

  if ((IsDelimitedEscapeSequence || IsNamedEscapeSequence)) {
    Diags_Report(Loc,diag::ext_delimited_escape_sequence, "");
  }

  return true;
}

static bool IsEscapeValidInUnevaluatedStringLiteral(char Escape) {
  switch (Escape) {
  case '\'':
  case '"':
  case '?':
  case '\\':
  case 'a':
  case 'b':
  case 'f':
  case 'n':
  case 'r':
  case 't':
  case 'v':
    return true;
  }
  return false;
}

/// ProcessCharEscape - Parse a standard C escape sequence, which can occur in
/// either a character or a string literal.
static unsigned ProcessCharEscape(const char *ThisTokBegin,
                                  const char *&ThisTokBuf,
                                  const char *ThisTokEnd, bool &HadError,
                                  uint32_t Loc, unsigned CharWidth,
                                  StringLiteralEvalMethod EvalMethod) {
  const char *EscapeBegin = ThisTokBuf;
  bool Delimited = false;
  bool EndDelimiterFound = false;

  // Skip the '\' char.
  ++ThisTokBuf;

  // We know that this character can't be off the end of the buffer, because
  // that would have been \", which would not have been the end of string.
  unsigned ResultChar = *ThisTokBuf++;
  char Escape = ResultChar;
  switch (ResultChar) {
  // These map to themselves.
  case '\\': case '\'': case '"': case '?': break;

    // These have fixed mappings.
  case 'a':
    // TODO: K&R: the meaning of '\\a' is different in traditional C
    ResultChar = 7;
    break;
  case 'b':
    ResultChar = 8;
    break;
  case 'e':
    Diags_Report(Loc,
                 diag::ext_nonstandard_escape, "e");
    ResultChar = 27;
    break;
  case 'E':
    Diags_Report(Loc,
                 diag::ext_nonstandard_escape, "E");
    ResultChar = 27;
    break;
  case 'f':
    ResultChar = 12;
    break;
  case 'n':
    ResultChar = 10;
    break;
  case 'r':
    ResultChar = 13;
    break;
  case 't':
    ResultChar = 9;
    break;
  case 'v':
    ResultChar = 11;
    break;
  case 'x': { // Hex escape.
    ResultChar = 0;
    if (ThisTokBuf != ThisTokEnd && *ThisTokBuf == '{') {
      Delimited = true;
      ThisTokBuf++;
      if (*ThisTokBuf == '}') {
        HadError = true;
        Diags_Report(Loc,
                     diag::err_delimited_escape_empty, "");
      }
    } else if (ThisTokBuf == ThisTokEnd || !isHexDigit(*ThisTokBuf)) {
      Diags_Report(Loc,
                   diag::err_hex_escape_no_digits, "x");
      return ResultChar;
    }

    // Hex escapes are a maximal series of hex digits.
    bool Overflow = false;
    for (; ThisTokBuf != ThisTokEnd; ++ThisTokBuf) {
      if (Delimited && *ThisTokBuf == '}') {
        ThisTokBuf++;
        EndDelimiterFound = true;
        break;
      }
      int CharVal = llvm::hexDigitValue(*ThisTokBuf);
      if (CharVal == -1) {
        // Non delimited hex escape sequences stop at the first non-hex digit.
        if (!Delimited)
          break;
        HadError = true;
        Diags_Report(Loc,
                     diag::err_delimited_escape_invalid,
                     llvm::StringRef(ThisTokBuf, 1).str());
        continue;
      }
      // About to shift out a digit?
      if (ResultChar & 0xF0000000)
        Overflow = true;
      ResultChar <<= 4;
      ResultChar |= CharVal;
    }
    // See if any bits will be truncated when evaluated as a character.
    if (CharWidth != 32 && (ResultChar >> CharWidth) != 0) {
      Overflow = true;
      ResultChar &= ~0U >> (32-CharWidth);
    }

    // Check for overflow.
    if (!HadError && Overflow) { // Too many digits to fit in
      HadError = true;
      Diags_Report(Loc,
                   diag::err_escape_too_large, "");
    }
    break;
  }
  case '0': case '1': case '2': case '3':
  case '4': case '5': case '6': case '7': {
    // Octal escapes.
    --ThisTokBuf;
    ResultChar = 0;

    // Octal escapes are a series of octal digits with maximum length 3.
    // "\0123" is a two digit sequence equal to "\012" "3".
    unsigned NumDigits = 0;
    do {
      ResultChar <<= 3;
      ResultChar |= *ThisTokBuf++ - '0';
      ++NumDigits;
    } while (ThisTokBuf != ThisTokEnd && NumDigits < 3 &&
             ThisTokBuf[0] >= '0' && ThisTokBuf[0] <= '7');

    // Check for overflow.  Reject '\777', but not L'\777'.
    if (CharWidth != 32 && (ResultChar >> CharWidth) != 0) {
      Diags_Report(Loc,
                   diag::err_escape_too_large, "1");
      ResultChar &= ~0U >> (32-CharWidth);
    }
    break;
  }
  case 'o': {
    bool Overflow = false;
    if (ThisTokBuf == ThisTokEnd || *ThisTokBuf != '{') {
      HadError = true;
      Diags_Report(Loc,
                   diag::err_delimited_escape_missing_brace,
                   "o");

      break;
    }
    ResultChar = 0;
    Delimited = true;
    ++ThisTokBuf;
    if (*ThisTokBuf == '}') {
      HadError = true;
      Diags_Report(Loc,
                   diag::err_delimited_escape_empty, "");
    }

    while (ThisTokBuf != ThisTokEnd) {
      if (*ThisTokBuf == '}') {
        EndDelimiterFound = true;
        ThisTokBuf++;
        break;
      }
      if (*ThisTokBuf < '0' || *ThisTokBuf > '7') {
        HadError = true;
        Diags_Report(Loc,
                     diag::err_delimited_escape_invalid,
                     llvm::StringRef(ThisTokBuf, 1).str());
        ThisTokBuf++;
        continue;
      }
      // Check if one of the top three bits is set before shifting them out.
      if (ResultChar & 0xE0000000)
        Overflow = true;

      ResultChar <<= 3;
      ResultChar |= *ThisTokBuf++ - '0';
    }
    // Check for overflow.  Reject '\777', but not L'\777'.
    if (!HadError &&
        (Overflow || (CharWidth != 32 && (ResultChar >> CharWidth) != 0))) {
      HadError = true;
      Diags_Report(Loc,
                   diag::err_escape_too_large, "");
      ResultChar &= ~0U >> (32 - CharWidth);
    }
    break;
  }
    // Otherwise, these are not valid escapes.
  case '(': case '{': case '[': case '%':
    // GCC accepts these as extensions.  We warn about them as such though.
    Diags_Report(Loc,
                 diag::ext_nonstandard_escape,
                 std::string(1, ResultChar));
    break;
  default:
    if (isPrintable(ResultChar))
      Diags_Report(Loc,
                   diag::ext_unknown_escape,
                   std::string(1, ResultChar));
    else
      Diags_Report(Loc,
                   diag::ext_unknown_escape,
                   "x" + llvm::utohexstr(ResultChar));
    break;
  }

  if (Delimited) {
    if (!EndDelimiterFound)
      Diags_Report(Loc,
                   diag::err_expected, "dil::TokenKind::r_brace");
    else if (!HadError) {
      Diags_Report(Loc,
                   diag::ext_delimited_escape_sequence, "");
    }
  }

  if (EvalMethod == StringLiteralEvalMethod::Unevaluated &&
      !IsEscapeValidInUnevaluatedStringLiteral(Escape)) {
    Diags_Report(Loc,
                 diag::err_unevaluated_string_invalid_escape_sequence,
                 llvm::StringRef(EscapeBegin, ThisTokBuf - EscapeBegin).str());
    HadError = true;
  }

  return ResultChar;
}

CharLiteralParser::CharLiteralParser(const char *begin, const char *end,
                                     unsigned Loc, DILLexer &lexer,
                                     dil::TokenKind kind) {
  // At this point we know that the character matches the regex "(L|u|U)?'.*'".
  HadError = false;

  Kind = kind;

  const char *TokBegin = begin;

  // Skip over wide character determinant.
  if (Kind != dil::TokenKind::char_constant)
    ++begin;
  if (Kind == dil::TokenKind::utf8_char_constant)
    ++begin;

  // Skip over the entry quote.
  if (begin[0] != '\'') {
    Diags_Report(Loc, diag::err_lexing_char, "");
    HadError = true;
    return;
  }

  ++begin;

  // Trim the ending quote.
  assert(end != begin && "Invalid token lexed");
  --end;

  // FIXME: The "Value" is an uint64_t so we can handle char literals of
  // up to 64-bits.
  // FIXME: This extensively assumes that 'char' is 8-bits.
  assert(lexer.getTargetInfo().getCharWidth() == 8 &&
         "Assumes char is 8 bits");
  assert(lexer.getTargetInfo().getIntWidth() <= 64 &&
             (lexer.getTargetInfo().getIntWidth() & 7) == 0 &&
         "Assumes sizeof(int) on target is <= 64 and a multiple of char");
  assert(lexer.getTargetInfo().getWCharWidth() <= 64 &&
         "Assumes sizeof(wchar) on target is <= 64");

  llvm::SmallVector<uint32_t, 4> codepoint_buffer;
  codepoint_buffer.resize(end - begin);
  uint32_t *buffer_begin = &codepoint_buffer.front();
  uint32_t *buffer_end = buffer_begin + codepoint_buffer.size();

  // Unicode escapes representing characters that cannot be correctly
  // represented in a single code unit are disallowed in character literals
  // by this implementation.
  uint32_t largest_character_for_kind;
  if (dil::TokenKind::wide_char_constant == Kind) {
    largest_character_for_kind =
        0xFFFFFFFFu >> (lexer.getTargetInfo().getWCharWidth());
  } else if (dil::TokenKind::utf8_char_constant == Kind) {
    largest_character_for_kind = 0x7F;
  } else {
    largest_character_for_kind = 0x7Fu;
  }

  while (begin != end) {
    // Is this a span of non-escape characters?
    if (begin[0] != '\\') {
      char const *start = begin;
      do {
        ++begin;
      } while (begin != end && *begin != '\\');

      char const *tmp_in_start = start;
      uint32_t *tmp_out_start = buffer_begin;
      llvm::ConversionResult res =
          llvm::ConvertUTF8toUTF32(reinterpret_cast<llvm::UTF8 const **>(&start),
                             reinterpret_cast<llvm::UTF8 const *>(begin),
                             &buffer_begin, buffer_end, llvm::strictConversion);
      if (res != llvm::conversionOK) {
        // If we see bad encoding for unprefixed character literals, warn and
        // simply copy the byte values, for compatibility with gcc and
        // older versions of clang.
        bool NoErrorOnBadEncoding = isOrdinary();
        dil::diag Msg = diag::err_bad_character_encoding;
        if (NoErrorOnBadEncoding)
          Msg = diag::warn_bad_character_encoding;
        Diags_Report(Loc, Msg, "");
        if (NoErrorOnBadEncoding) {
          start = tmp_in_start;
          buffer_begin = tmp_out_start;
          for (; start != begin; ++start, ++buffer_begin)
            *buffer_begin = static_cast<uint8_t>(*start);
        } else {
          HadError = true;
        }
      } else {
        for (; tmp_out_start < buffer_begin; ++tmp_out_start) {
          if (*tmp_out_start > largest_character_for_kind) {
            HadError = true;
            Diags_Report(Loc, diag::err_character_too_large,
                         "");
          }
        }
      }

      continue;
    }
    // Is this a Universal Character Name escape?
    if (begin[1] == 'u' || begin[1] == 'U' || begin[1] == 'N') {
      unsigned short UcnLen = 0;
      if (!ProcessUCNEscape(TokBegin, begin, end, *buffer_begin, UcnLen,
                            Loc, true)) {
        HadError = true;
      } else if (*buffer_begin > largest_character_for_kind) {
        HadError = true;
        Diags_Report(Loc, diag::err_character_too_large, "");
      }

      ++buffer_begin;
      continue;
    }
    unsigned CharWidth = getCharWidth(Kind, lexer.getTargetInfo());
    uint64_t result =
        ProcessCharEscape(TokBegin, begin, end, HadError, Loc,CharWidth,
                          StringLiteralEvalMethod::Evaluated);
    *buffer_begin++ = result;
  }

  unsigned NumCharsSoFar = buffer_begin - &codepoint_buffer.front();

  if (NumCharsSoFar > 1) {
    if (isOrdinary() && NumCharsSoFar == 4)
      Diags_Report(Loc, diag::warn_four_char_character_literal, "");
    else if (isOrdinary())
      Diags_Report(Loc, diag::warn_multichar_character_literal, "");
    else {
      Diags_Report(Loc, diag::err_multichar_character_literal,
                   isWide() ? "0" : "1");
      HadError = true;
    }
    IsMultiChar = true;
  } else {
    IsMultiChar = false;
  }

  llvm::APInt LitVal(lexer.getTargetInfo().getIntWidth(), 0);

  // Narrow character literals act as though their value is concatenated
  // in this implementation, but warn on overflow.
  bool multi_char_too_long = false;
  if (isOrdinary() && isMultiChar()) {
    LitVal = 0;
    for (size_t i = 0; i < NumCharsSoFar; ++i) {
      // check for enough leading zeros to shift into
      multi_char_too_long |= (LitVal.countl_zero() < 8);
      LitVal <<= 8;
      LitVal = LitVal + (codepoint_buffer[i] & 0xFF);
    }
  } else if (NumCharsSoFar > 0) {
    // otherwise just take the last character
    LitVal = buffer_begin[-1];
  }

  if (!HadError && multi_char_too_long) {
    Diags_Report(Loc, diag::warn_char_constant_too_large, "");
  }

  // Transfer the value from APInt to uint64_t
  Value = LitVal.getZExtValue();

  // If this is a single narrow character, sign extend it (e.g. '\xFF' is "-1")
  // if 'char' is signed for this target (C99 6.4.4.4p10).  Note that multiple
  // character constants are not sign extended in the this implementation:
  // '\xFF\xFF' = 65536 and '\x0\xFF' = 255, which matches GCC.
  if (isOrdinary() && NumCharsSoFar == 1 && (Value & 128))
    Value = (signed char)Value;
}


StringLiteralParser::StringLiteralParser(llvm::ArrayRef<DILToken> StringToks,
                                         DILLexer &lexer,
                                         StringLiteralEvalMethod EvalMethod) :
    MaxTokenLength(0), SizeBound(0), CharByteWidth(0),
    Kind(dil::TokenKind::unknown), ResultPtr(ResultBuf.data()),
    EvalMethod(EvalMethod), m_lexer(lexer), hadError(false) {
  init(StringToks, lexer);
}


void StringLiteralParser::init(llvm::ArrayRef<DILToken> StringToks,
                               DILLexer &lexer) {
  // The literal token may have come from an invalid source location (e.g. due
  // to a PCH error), in which case the token length will be 0.
  if (StringToks.empty() || StringToks[0].getLength() < 2)
    return DiagnoseLexingError(lexer.GetLocation());

  // Scan all of the string portions, remember the max individual token length,
  // computing a bound on the concatenated string length, and see whether any
  // piece is a wide-string.  If any of the string portions is a wide-string
  // literal, the result is a wide-string literal [C99 6.4.5p4].
  assert(!StringToks.empty() && "expected at least one token");
  MaxTokenLength = StringToks[0].getLength();
  assert(StringToks[0].getLength() >= 2 && "literal token is invalid!");
  SizeBound = StringToks[0].getLength() - 2; // -2 for "".
  hadError = false;

  // Determines the kind of string from the prefix
  Kind = dil::TokenKind::string_literal;

  /// (C99 5.1.1.2p1).  The common case is only one string fragment.
  for (const DILToken &Tok : StringToks) {
    if (Tok.getLength() < 2)
      return DiagnoseLexingError(Tok.getLocation());

    // The string could be shorter than this if it needs cleaning, but this is a
    // reasonable bound, which is all we need.
    assert(Tok.getLength() >= 2 && "literal token is invalid!");
    SizeBound += Tok.getLength() - 2; // -2 for "".

    // Remember maximum string piece length.
    if (Tok.getLength() > MaxTokenLength)
      MaxTokenLength = Tok.getLength();

    // Remember if we see any wide or utf-8/16/32 strings.
    // Also check for illegal concatenations.
    if (isUnevaluated() && Tok.getKind() != dil::TokenKind::string_literal) {
      Diags_Report(Tok.getLocation(),
                   diag::warn_unevaluated_string_prefix, "");
        hadError = true;
    } else if (Tok.isNot(Kind) && Tok.isNot(dil::TokenKind::string_literal)) {
      if (isOrdinary()) {
        Kind = Tok.getKind();
      } else {
        Diags_Report(Tok.getLocation(), diag::err_unsupported_string_concat,
                     "");
        hadError = true;
      }
    }
  }

  // Include space for the null terminator.
  ++SizeBound;

  // TODO: K&R warning: "traditional C rejects string constant concatenation"

  // Get the width in bytes of char/wchar_t/char16_t/char32_t
  CharByteWidth = getCharWidth(Kind, lexer.getTargetInfo());
  assert((CharByteWidth & 7) == 0 && "Assumes character size is byte multiple");
  CharByteWidth /= 8;

  // The output buffer size needs to be large enough to hold wide characters.
  // This is a worst-case assumption which basically corresponds to L"" "long".
  SizeBound *= CharByteWidth;

  // Size the temporary buffer to hold the result string data.
  ResultBuf.resize(SizeBound);

  // Likewise, but for each string piece.
  llvm::SmallString<512> TokenBuf;
  TokenBuf.resize(MaxTokenLength);

  // Loop over all the strings, getting their spelling, and expanding them to
  // wide strings as appropriate.
  ResultPtr = &ResultBuf[0];   // Next byte to fill in.

  for (unsigned i = 0, e = StringToks.size(); i != e; ++i) {
    const char *ThisTokBuf = &TokenBuf[0];
    // Get the spelling of the token, which eliminates trigraphs, etc.  We know
    // that ThisTokBuf points to a buffer that is big enough for the whole token
    // and 'spelled' tokens can only shrink.
    bool StringInvalid = false;
    //unsigned ThisTokLen = // CAROLINE!!
    //  Lexer::getSpelling(StringToks[i], ThisTokBuf, SM, Features,
    //                     &StringInvalid);
    ThisTokBuf = StringToks[i].getSpelling().data();
    unsigned ThisTokLen = StringToks[i].getLength();
    if (StringInvalid)
      return DiagnoseLexingError(StringToks[i].getLocation());

    const char *ThisTokBegin = ThisTokBuf;
    const char *ThisTokEnd = ThisTokBuf+ThisTokLen;

    // Strip the end quote.
    --ThisTokEnd;

    // TODO: Input character set mapping support.

    // Skip marker for wide or unicode strings.
    if (ThisTokBuf[0] == 'L' || ThisTokBuf[0] == 'u' || ThisTokBuf[0] == 'U') {
      ++ThisTokBuf;
      // Skip 8 of u8 marker for utf8 strings.
      if (ThisTokBuf[0] == '8')
        ++ThisTokBuf;
    }

    // Check for raw string
    if (ThisTokBuf[0] == 'R') {
      if (ThisTokBuf[1] != '"') {
        // The file may have come from PCH and then changed after loading the
        // PCH; Fail gracefully.
        return DiagnoseLexingError(StringToks[i].getLocation());
      }
      ThisTokBuf += 2; // skip R"

      // C++11 [lex.string]p2: A `d-char-sequence` shall consist of at most 16
      // characters.
      constexpr unsigned MaxRawStrDelimLen = 16;

      const char *Prefix = ThisTokBuf;
      while (static_cast<unsigned>(ThisTokBuf - Prefix) < MaxRawStrDelimLen &&
             ThisTokBuf[0] != '(')
        ++ThisTokBuf;
      if (ThisTokBuf[0] != '(')
        return DiagnoseLexingError(StringToks[i].getLocation());
      ++ThisTokBuf; // skip '('

      // Remove same number of characters from the end
      ThisTokEnd -= ThisTokBuf - Prefix;
      if (ThisTokEnd < ThisTokBuf)
        return DiagnoseLexingError(StringToks[i].getLocation());

      // C++14 [lex.string]p4: A source-file new-line in a raw string literal
      // results in a new-line in the resulting execution string-literal.
      llvm::StringRef RemainingTokenSpan(ThisTokBuf, ThisTokEnd - ThisTokBuf);
      while (!RemainingTokenSpan.empty()) {
        // Split the string literal on \r\n boundaries.
        size_t CRLFPos = RemainingTokenSpan.find("\r\n");
        llvm::StringRef BeforeCRLF = RemainingTokenSpan.substr(0, CRLFPos);
        llvm::StringRef AfterCRLF = RemainingTokenSpan.substr(CRLFPos);

        // Copy everything before the \r\n sequence into the string literal.
        if (CopyStringFragment(StringToks[i], ThisTokBegin, BeforeCRLF))
          hadError = true;

        // Point into the \n inside the \r\n sequence and operate on the
        // remaining portion of the literal.
        RemainingTokenSpan = AfterCRLF.substr(1);
      }
    } else {
      if (ThisTokBuf[0] != '"') {
        // The file may have come from PCH and then changed after loading the
        // PCH; Fail gracefully.
        return DiagnoseLexingError(StringToks[i].getLocation());
      }
      ++ThisTokBuf; // skip "

      while (ThisTokBuf != ThisTokEnd) {
        // Is this a span of non-escape characters?
        if (ThisTokBuf[0] != '\\') {
          const char *InStart = ThisTokBuf;
          do {
            ++ThisTokBuf;
          } while (ThisTokBuf != ThisTokEnd && ThisTokBuf[0] != '\\');

          // Copy the character span over.
          if (CopyStringFragment(StringToks[i], ThisTokBegin,
                                 llvm::StringRef(InStart, ThisTokBuf - InStart)))
            hadError = true;
          continue;
        }
        // Is this a Universal Character Name escape?
        if (ThisTokBuf[1] == 'u' || ThisTokBuf[1] == 'U' ||
            ThisTokBuf[1] == 'N') {
          EncodeUCNEscape(ThisTokBegin, ThisTokBuf, ThisTokEnd,
                          ResultPtr, hadError,
                          StringToks[i].getLocation(),
                          CharByteWidth);
          continue;
        }
        // Otherwise, this is a non-UCN escape character.  Process it.
        unsigned ResultChar =
            ProcessCharEscape(ThisTokBegin, ThisTokBuf, ThisTokEnd, hadError,
                              StringToks[i].getLocation(),
                              CharByteWidth * 8, EvalMethod);

        if (CharByteWidth == 4) {
          // FIXME: Make the type of the result buffer correct instead of
          // using reinterpret_cast.
          llvm::UTF32 *ResultWidePtr = reinterpret_cast<llvm::UTF32*>(ResultPtr);
          *ResultWidePtr = ResultChar;
          ResultPtr += 4;
        } else if (CharByteWidth == 2) {
          // FIXME: Make the type of the result buffer correct instead of
          // using reinterpret_cast.
          llvm::UTF16 *ResultWidePtr = reinterpret_cast<llvm::UTF16*>(ResultPtr);
          *ResultWidePtr = ResultChar & 0xFFFF;
          ResultPtr += 2;
        } else {
          assert(CharByteWidth == 1 && "Unexpected char width");
          *ResultPtr++ = ResultChar & 0xFF;
        }
      }
    }
  }

  // Complain if this string literal has too many characters.
  unsigned MaxChars = 65536;

  if (GetNumStringChars() > MaxChars)
    Diags_Report(StringToks.front().getLocation(),
                 diag::ext_string_too_long,"");
}

static const char *resyncUTF8(const char *Err, const char *End) {
  if (Err == End)
    return End;
  End = Err + std::min<unsigned>(llvm::getNumBytesForUTF8(*Err), End-Err);
  while (++Err != End && (*Err & 0xC0) == 0x80)
    ;
  return Err;
}


/// MeasureUCNEscape - Determine the number of bytes within the resulting string
/// which this UCN will occupy.
static int MeasureUCNEscape(const char *ThisTokBegin, const char *&ThisTokBuf,
                            const char *ThisTokEnd, unsigned CharByteWidth,
                            bool &HadError) {
  // UTF-32: 4 bytes per escape.
  if (CharByteWidth == 4)
    return 4;

  uint32_t UcnVal = 0;
  unsigned short UcnLen = 0;
  uint32_t Loc;

  if (!ProcessUCNEscape(ThisTokBegin, ThisTokBuf, ThisTokEnd, UcnVal,
                        UcnLen, Loc, true)) {
    HadError = true;
    return 0;
  }

  // UTF-16: 2 bytes for BMP, 4 bytes otherwise.
  if (CharByteWidth == 2)
    return UcnVal <= 0xFFFF ? 2 : 4;

  // UTF-8.
  if (UcnVal < 0x80)
    return 1;
  if (UcnVal < 0x800)
    return 2;
  if (UcnVal < 0x10000)
    return 3;
  return 4;
}


unsigned StringLiteralParser::getOffsetOfStringByte(const DILToken &Tok,
                                                    unsigned ByteNo) const {
  // Get the spelling of the token.
  llvm::SmallString<32> SpellingBuffer;
  SpellingBuffer.resize(Tok.getLength());

  bool StringInvalid = false;
  const char *SpellingPtr = &SpellingBuffer[0];
  SpellingPtr = Tok.getSpelling().data();
  unsigned TokLen = Tok.getLength();
  //unsigned TokLen = Lexer::getSpelling(Tok, SpellingPtr, SM, Features,
  //           &StringInvalid);
  if (StringInvalid)
    return 0;

  const char *SpellingStart = SpellingPtr;
  const char *SpellingEnd = SpellingPtr+TokLen;

  // Handle UTF-8 strings just like narrow strings.
  if (SpellingPtr[0] == 'u' && SpellingPtr[1] == '8')
    SpellingPtr += 2;

  assert(SpellingPtr[0] != 'L' && SpellingPtr[0] != 'u' &&
         SpellingPtr[0] != 'U' && "Doesn't handle wide or utf strings yet");

  // For raw string literals, this is easy.
  if (SpellingPtr[0] == 'R') {
    assert(SpellingPtr[1] == '"' && "Should be a raw string literal!");
    // Skip 'R"'.
    SpellingPtr += 2;
    while (*SpellingPtr != '(') {
      ++SpellingPtr;
      assert(SpellingPtr < SpellingEnd && "Missing ( for raw string literal");
    }
    // Skip '('.
    ++SpellingPtr;
    return SpellingPtr - SpellingStart + ByteNo;
  }

  // Skip over the leading quote
  assert(SpellingPtr[0] == '"' && "Should be a string literal!");
  ++SpellingPtr;

  // Skip over bytes until we find the offset we're looking for.
  while (ByteNo) {
    assert(SpellingPtr < SpellingEnd && "Didn't find byte offset!");

    // Step over non-escapes simply.
    if (*SpellingPtr != '\\') {
      ++SpellingPtr;
      --ByteNo;
      continue;
    }

    // Otherwise, this is an escape character.  Advance over it.
    bool HadError = false;
    if (SpellingPtr[1] == 'u' || SpellingPtr[1] == 'U' ||
        SpellingPtr[1] == 'N') {
      const char *EscapePtr = SpellingPtr;
      unsigned Len = MeasureUCNEscape(SpellingStart, SpellingPtr, SpellingEnd,
                                      1, HadError);
      if (Len > ByteNo) {
        // ByteNo is somewhere within the escape sequence.
        SpellingPtr = EscapePtr;
        break;
      }
      ByteNo -= Len;
    } else {
      ProcessCharEscape(SpellingStart, SpellingPtr, SpellingEnd, HadError,
                        Tok.getLocation(), CharByteWidth * 8,
                        StringLiteralEvalMethod::Evaluated);
      --ByteNo;
    }
    assert(!HadError && "This method isn't valid on erroneous strings");
  }

  return SpellingPtr-SpellingStart;
}


bool StringLiteralParser::CopyStringFragment(const DILToken &Tok,
                                             const char *TokBegin,
                                             llvm::StringRef Fragment) {
  const llvm::UTF8 *ErrorPtrTmp;
  if (ConvertUTF8toWide(CharByteWidth, Fragment, ResultPtr, ErrorPtrTmp))
    return false;

  // If we see bad encoding for unprefixed string literals, warn and
  // simply copy the byte values, for compatibility with gcc and older
  // versions of clang.
  bool NoErrorOnBadEncoding = isOrdinary();
  if (NoErrorOnBadEncoding) {
    memcpy(ResultPtr, Fragment.data(), Fragment.size());
    ResultPtr += Fragment.size();
  }

  const char *ErrorPtr = reinterpret_cast<const char *>(ErrorPtrTmp);

  uint32_t SourceLoc = Tok.getLocation();
  Diags_Report(SourceLoc,
               NoErrorOnBadEncoding ? diag::warn_bad_string_encoding
               : diag::err_bad_string_encoding, "");

  const char *NextStart = resyncUTF8(ErrorPtr, Fragment.end());
  llvm::StringRef NextFragment(NextStart, Fragment.end()-NextStart);

  // Decode into a dummy buffer.
  llvm::SmallString<512> Dummy;
  Dummy.reserve(Fragment.size() * CharByteWidth);
  char *Ptr = Dummy.data();

  while (!ConvertUTF8toWide(CharByteWidth, NextFragment, Ptr, ErrorPtrTmp)) {
    const char *ErrorPtr = reinterpret_cast<const char *>(ErrorPtrTmp);
    NextStart = resyncUTF8(ErrorPtr, Fragment.end());
    NextFragment = llvm::StringRef(NextStart, Fragment.end()-NextStart);
  }

  return !NoErrorOnBadEncoding;
}


void StringLiteralParser::DiagnoseLexingError(unsigned Loc) {
  hadError = true;
  Diags_Report(Loc, diag::err_lexing_string, "");
}

} // namespace dil

} // namespace lldb_private
