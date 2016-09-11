//===- MILexer.cpp - Machine instructions lexer implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the lexing of machine instructions.
//
//===----------------------------------------------------------------------===//

#include "MILexer.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include <cctype>

using namespace llvm;

namespace {

typedef function_ref<void(StringRef::iterator Loc, const Twine &)>
    ErrorCallbackType;

/// This class provides a way to iterate and get characters from the source
/// string.
class Cursor {
  const char *Ptr;
  const char *End;

public:
  Cursor(NoneType) : Ptr(nullptr), End(nullptr) {}

  explicit Cursor(StringRef Str) {
    Ptr = Str.data();
    End = Ptr + Str.size();
  }

  bool isEOF() const { return Ptr == End; }

  char peek(int I = 0) const { return End - Ptr <= I ? 0 : Ptr[I]; }

  void advance(unsigned I = 1) { Ptr += I; }

  StringRef remaining() const { return StringRef(Ptr, End - Ptr); }

  StringRef upto(Cursor C) const {
    assert(C.Ptr >= Ptr && C.Ptr <= End);
    return StringRef(Ptr, C.Ptr - Ptr);
  }

  StringRef::iterator location() const { return Ptr; }

  operator bool() const { return Ptr != nullptr; }
};

} // end anonymous namespace

MIToken &MIToken::reset(TokenKind Kind, StringRef Range) {
  this->Kind = Kind;
  this->Range = Range;
  return *this;
}

MIToken &MIToken::setStringValue(StringRef StrVal) {
  StringValue = StrVal;
  return *this;
}

MIToken &MIToken::setOwnedStringValue(std::string StrVal) {
  StringValueStorage = std::move(StrVal);
  StringValue = StringValueStorage;
  return *this;
}

MIToken &MIToken::setIntegerValue(APSInt IntVal) {
  this->IntVal = std::move(IntVal);
  return *this;
}

/// Skip the leading whitespace characters and return the updated cursor.
static Cursor skipWhitespace(Cursor C) {
  while (isblank(C.peek()))
    C.advance();
  return C;
}

static bool isNewlineChar(char C) { return C == '\n' || C == '\r'; }

/// Skip a line comment and return the updated cursor.
static Cursor skipComment(Cursor C) {
  if (C.peek() != ';')
    return C;
  while (!isNewlineChar(C.peek()) && !C.isEOF())
    C.advance();
  return C;
}

/// Return true if the given character satisfies the following regular
/// expression: [-a-zA-Z$._0-9]
static bool isIdentifierChar(char C) {
  return isalpha(C) || isdigit(C) || C == '_' || C == '-' || C == '.' ||
         C == '$';
}

/// Unescapes the given string value.
///
/// Expects the string value to be quoted.
static std::string unescapeQuotedString(StringRef Value) {
  assert(Value.front() == '"' && Value.back() == '"');
  Cursor C = Cursor(Value.substr(1, Value.size() - 2));

  std::string Str;
  Str.reserve(C.remaining().size());
  while (!C.isEOF()) {
    char Char = C.peek();
    if (Char == '\\') {
      if (C.peek(1) == '\\') {
        // Two '\' become one
        Str += '\\';
        C.advance(2);
        continue;
      }
      if (isxdigit(C.peek(1)) && isxdigit(C.peek(2))) {
        Str += hexDigitValue(C.peek(1)) * 16 + hexDigitValue(C.peek(2));
        C.advance(3);
        continue;
      }
    }
    Str += Char;
    C.advance();
  }
  return Str;
}

/// Lex a string constant using the following regular expression: \"[^\"]*\"
static Cursor lexStringConstant(Cursor C, ErrorCallbackType ErrorCallback) {
  assert(C.peek() == '"');
  for (C.advance(); C.peek() != '"'; C.advance()) {
    if (C.isEOF() || isNewlineChar(C.peek())) {
      ErrorCallback(
          C.location(),
          "end of machine instruction reached before the closing '\"'");
      return None;
    }
  }
  C.advance();
  return C;
}

static Cursor lexName(Cursor C, MIToken &Token, MIToken::TokenKind Type,
                      unsigned PrefixLength, ErrorCallbackType ErrorCallback) {
  auto Range = C;
  C.advance(PrefixLength);
  if (C.peek() == '"') {
    if (Cursor R = lexStringConstant(C, ErrorCallback)) {
      StringRef String = Range.upto(R);
      Token.reset(Type, String)
          .setOwnedStringValue(
              unescapeQuotedString(String.drop_front(PrefixLength)));
      return R;
    }
    Token.reset(MIToken::Error, Range.remaining());
    return Range;
  }
  while (isIdentifierChar(C.peek()))
    C.advance();
  Token.reset(Type, Range.upto(C))
      .setStringValue(Range.upto(C).drop_front(PrefixLength));
  return C;
}

static Cursor maybeLexIntegerOrScalarType(Cursor C, MIToken &Token) {
  if ((C.peek() != 'i' && C.peek() != 's' && C.peek() != 'p') ||
      !isdigit(C.peek(1)))
    return None;
  char Kind = C.peek();
  auto Range = C;
  C.advance(); // Skip 'i', 's', or 'p'
  while (isdigit(C.peek()))
    C.advance();

  Token.reset(Kind == 'i'
                  ? MIToken::IntegerType
                  : (Kind == 's' ? MIToken::ScalarType : MIToken::PointerType),
              Range.upto(C));
  return C;
}

static MIToken::TokenKind getIdentifierKind(StringRef Identifier) {
  return StringSwitch<MIToken::TokenKind>(Identifier)
      .Case("_", MIToken::underscore)
      .Case("implicit", MIToken::kw_implicit)
      .Case("implicit-def", MIToken::kw_implicit_define)
      .Case("def", MIToken::kw_def)
      .Case("dead", MIToken::kw_dead)
      .Case("killed", MIToken::kw_killed)
      .Case("undef", MIToken::kw_undef)
      .Case("internal", MIToken::kw_internal)
      .Case("early-clobber", MIToken::kw_early_clobber)
      .Case("debug-use", MIToken::kw_debug_use)
      .Case("tied-def", MIToken::kw_tied_def)
      .Case("frame-setup", MIToken::kw_frame_setup)
      .Case("debug-location", MIToken::kw_debug_location)
      .Case("same_value", MIToken::kw_cfi_same_value)
      .Case("offset", MIToken::kw_cfi_offset)
      .Case("def_cfa_register", MIToken::kw_cfi_def_cfa_register)
      .Case("def_cfa_offset", MIToken::kw_cfi_def_cfa_offset)
      .Case("def_cfa", MIToken::kw_cfi_def_cfa)
      .Case("blockaddress", MIToken::kw_blockaddress)
      .Case("intrinsic", MIToken::kw_intrinsic)
      .Case("target-index", MIToken::kw_target_index)
      .Case("half", MIToken::kw_half)
      .Case("float", MIToken::kw_float)
      .Case("double", MIToken::kw_double)
      .Case("x86_fp80", MIToken::kw_x86_fp80)
      .Case("fp128", MIToken::kw_fp128)
      .Case("ppc_fp128", MIToken::kw_ppc_fp128)
      .Case("target-flags", MIToken::kw_target_flags)
      .Case("volatile", MIToken::kw_volatile)
      .Case("non-temporal", MIToken::kw_non_temporal)
      .Case("dereferenceable", MIToken::kw_dereferenceable)
      .Case("invariant", MIToken::kw_invariant)
      .Case("align", MIToken::kw_align)
      .Case("stack", MIToken::kw_stack)
      .Case("got", MIToken::kw_got)
      .Case("jump-table", MIToken::kw_jump_table)
      .Case("constant-pool", MIToken::kw_constant_pool)
      .Case("call-entry", MIToken::kw_call_entry)
      .Case("liveout", MIToken::kw_liveout)
      .Case("address-taken", MIToken::kw_address_taken)
      .Case("landing-pad", MIToken::kw_landing_pad)
      .Case("liveins", MIToken::kw_liveins)
      .Case("successors", MIToken::kw_successors)
      .Case("floatpred", MIToken::kw_floatpred)
      .Case("intpred", MIToken::kw_intpred)
      .Default(MIToken::Identifier);
}

static Cursor maybeLexIdentifier(Cursor C, MIToken &Token) {
  if (!isalpha(C.peek()) && C.peek() != '_')
    return None;
  auto Range = C;
  while (isIdentifierChar(C.peek()))
    C.advance();
  auto Identifier = Range.upto(C);
  Token.reset(getIdentifierKind(Identifier), Identifier)
      .setStringValue(Identifier);
  return C;
}

static Cursor maybeLexMachineBasicBlock(Cursor C, MIToken &Token,
                                        ErrorCallbackType ErrorCallback) {
  bool IsReference = C.remaining().startswith("%bb.");
  if (!IsReference && !C.remaining().startswith("bb."))
    return None;
  auto Range = C;
  unsigned PrefixLength = IsReference ? 4 : 3;
  C.advance(PrefixLength); // Skip '%bb.' or 'bb.'
  if (!isdigit(C.peek())) {
    Token.reset(MIToken::Error, C.remaining());
    ErrorCallback(C.location(), "expected a number after '%bb.'");
    return C;
  }
  auto NumberRange = C;
  while (isdigit(C.peek()))
    C.advance();
  StringRef Number = NumberRange.upto(C);
  unsigned StringOffset = PrefixLength + Number.size(); // Drop '%bb.<id>'
  if (C.peek() == '.') {
    C.advance(); // Skip '.'
    ++StringOffset;
    while (isIdentifierChar(C.peek()))
      C.advance();
  }
  Token.reset(IsReference ? MIToken::MachineBasicBlock
                          : MIToken::MachineBasicBlockLabel,
              Range.upto(C))
      .setIntegerValue(APSInt(Number))
      .setStringValue(Range.upto(C).drop_front(StringOffset));
  return C;
}

static Cursor maybeLexIndex(Cursor C, MIToken &Token, StringRef Rule,
                            MIToken::TokenKind Kind) {
  if (!C.remaining().startswith(Rule) || !isdigit(C.peek(Rule.size())))
    return None;
  auto Range = C;
  C.advance(Rule.size());
  auto NumberRange = C;
  while (isdigit(C.peek()))
    C.advance();
  Token.reset(Kind, Range.upto(C)).setIntegerValue(APSInt(NumberRange.upto(C)));
  return C;
}

static Cursor maybeLexIndexAndName(Cursor C, MIToken &Token, StringRef Rule,
                                   MIToken::TokenKind Kind) {
  if (!C.remaining().startswith(Rule) || !isdigit(C.peek(Rule.size())))
    return None;
  auto Range = C;
  C.advance(Rule.size());
  auto NumberRange = C;
  while (isdigit(C.peek()))
    C.advance();
  StringRef Number = NumberRange.upto(C);
  unsigned StringOffset = Rule.size() + Number.size();
  if (C.peek() == '.') {
    C.advance();
    ++StringOffset;
    while (isIdentifierChar(C.peek()))
      C.advance();
  }
  Token.reset(Kind, Range.upto(C))
      .setIntegerValue(APSInt(Number))
      .setStringValue(Range.upto(C).drop_front(StringOffset));
  return C;
}

static Cursor maybeLexJumpTableIndex(Cursor C, MIToken &Token) {
  return maybeLexIndex(C, Token, "%jump-table.", MIToken::JumpTableIndex);
}

static Cursor maybeLexStackObject(Cursor C, MIToken &Token) {
  return maybeLexIndexAndName(C, Token, "%stack.", MIToken::StackObject);
}

static Cursor maybeLexFixedStackObject(Cursor C, MIToken &Token) {
  return maybeLexIndex(C, Token, "%fixed-stack.", MIToken::FixedStackObject);
}

static Cursor maybeLexConstantPoolItem(Cursor C, MIToken &Token) {
  return maybeLexIndex(C, Token, "%const.", MIToken::ConstantPoolItem);
}

static Cursor maybeLexSubRegisterIndex(Cursor C, MIToken &Token,
                                       ErrorCallbackType ErrorCallback) {
  const StringRef Rule = "%subreg.";
  if (!C.remaining().startswith(Rule))
    return None;
  return lexName(C, Token, MIToken::SubRegisterIndex, Rule.size(),
                 ErrorCallback);
}

static Cursor maybeLexIRBlock(Cursor C, MIToken &Token,
                              ErrorCallbackType ErrorCallback) {
  const StringRef Rule = "%ir-block.";
  if (!C.remaining().startswith(Rule))
    return None;
  if (isdigit(C.peek(Rule.size())))
    return maybeLexIndex(C, Token, Rule, MIToken::IRBlock);
  return lexName(C, Token, MIToken::NamedIRBlock, Rule.size(), ErrorCallback);
}

static Cursor maybeLexIRValue(Cursor C, MIToken &Token,
                              ErrorCallbackType ErrorCallback) {
  const StringRef Rule = "%ir.";
  if (!C.remaining().startswith(Rule))
    return None;
  if (isdigit(C.peek(Rule.size())))
    return maybeLexIndex(C, Token, Rule, MIToken::IRValue);
  return lexName(C, Token, MIToken::NamedIRValue, Rule.size(), ErrorCallback);
}

static Cursor lexVirtualRegister(Cursor C, MIToken &Token) {
  auto Range = C;
  C.advance(); // Skip '%'
  auto NumberRange = C;
  while (isdigit(C.peek()))
    C.advance();
  Token.reset(MIToken::VirtualRegister, Range.upto(C))
      .setIntegerValue(APSInt(NumberRange.upto(C)));
  return C;
}

/// Returns true for a character allowed in a register name.
static bool isRegisterChar(char C) {
  return isIdentifierChar(C) && C != '.';
}

static Cursor maybeLexRegister(Cursor C, MIToken &Token) {
  if (C.peek() != '%')
    return None;
  if (isdigit(C.peek(1)))
    return lexVirtualRegister(C, Token);
  auto Range = C;
  C.advance(); // Skip '%'
  while (isRegisterChar(C.peek()))
    C.advance();
  Token.reset(MIToken::NamedRegister, Range.upto(C))
      .setStringValue(Range.upto(C).drop_front(1)); // Drop the '%'
  return C;
}

static Cursor maybeLexGlobalValue(Cursor C, MIToken &Token,
                                  ErrorCallbackType ErrorCallback) {
  if (C.peek() != '@')
    return None;
  if (!isdigit(C.peek(1)))
    return lexName(C, Token, MIToken::NamedGlobalValue, /*PrefixLength=*/1,
                   ErrorCallback);
  auto Range = C;
  C.advance(1); // Skip the '@'
  auto NumberRange = C;
  while (isdigit(C.peek()))
    C.advance();
  Token.reset(MIToken::GlobalValue, Range.upto(C))
      .setIntegerValue(APSInt(NumberRange.upto(C)));
  return C;
}

static Cursor maybeLexExternalSymbol(Cursor C, MIToken &Token,
                                     ErrorCallbackType ErrorCallback) {
  if (C.peek() != '$')
    return None;
  return lexName(C, Token, MIToken::ExternalSymbol, /*PrefixLength=*/1,
                 ErrorCallback);
}

static bool isValidHexFloatingPointPrefix(char C) {
  return C == 'H' || C == 'K' || C == 'L' || C == 'M';
}

static Cursor maybeLexHexFloatingPointLiteral(Cursor C, MIToken &Token) {
  if (C.peek() != '0' || C.peek(1) != 'x')
    return None;
  Cursor Range = C;
  C.advance(2); // Skip '0x'
  if (isValidHexFloatingPointPrefix(C.peek()))
    C.advance();
  while (isxdigit(C.peek()))
    C.advance();
  Token.reset(MIToken::FloatingPointLiteral, Range.upto(C));
  return C;
}

static Cursor lexFloatingPointLiteral(Cursor Range, Cursor C, MIToken &Token) {
  C.advance();
  // Skip over [0-9]*([eE][-+]?[0-9]+)?
  while (isdigit(C.peek()))
    C.advance();
  if ((C.peek() == 'e' || C.peek() == 'E') &&
      (isdigit(C.peek(1)) ||
       ((C.peek(1) == '-' || C.peek(1) == '+') && isdigit(C.peek(2))))) {
    C.advance(2);
    while (isdigit(C.peek()))
      C.advance();
  }
  Token.reset(MIToken::FloatingPointLiteral, Range.upto(C));
  return C;
}

static Cursor maybeLexNumericalLiteral(Cursor C, MIToken &Token) {
  if (!isdigit(C.peek()) && (C.peek() != '-' || !isdigit(C.peek(1))))
    return None;
  auto Range = C;
  C.advance();
  while (isdigit(C.peek()))
    C.advance();
  if (C.peek() == '.')
    return lexFloatingPointLiteral(Range, C, Token);
  StringRef StrVal = Range.upto(C);
  Token.reset(MIToken::IntegerLiteral, StrVal).setIntegerValue(APSInt(StrVal));
  return C;
}

static MIToken::TokenKind getMetadataKeywordKind(StringRef Identifier) {
  return StringSwitch<MIToken::TokenKind>(Identifier)
      .Case("!tbaa", MIToken::md_tbaa)
      .Case("!alias.scope", MIToken::md_alias_scope)
      .Case("!noalias", MIToken::md_noalias)
      .Case("!range", MIToken::md_range)
      .Default(MIToken::Error);
}

static Cursor maybeLexExlaim(Cursor C, MIToken &Token,
                             ErrorCallbackType ErrorCallback) {
  if (C.peek() != '!')
    return None;
  auto Range = C;
  C.advance(1);
  if (isdigit(C.peek()) || !isIdentifierChar(C.peek())) {
    Token.reset(MIToken::exclaim, Range.upto(C));
    return C;
  }
  while (isIdentifierChar(C.peek()))
    C.advance();
  StringRef StrVal = Range.upto(C);
  Token.reset(getMetadataKeywordKind(StrVal), StrVal);
  if (Token.isError())
    ErrorCallback(Token.location(),
                  "use of unknown metadata keyword '" + StrVal + "'");
  return C;
}

static MIToken::TokenKind symbolToken(char C) {
  switch (C) {
  case ',':
    return MIToken::comma;
  case '.':
    return MIToken::dot;
  case '=':
    return MIToken::equal;
  case ':':
    return MIToken::colon;
  case '(':
    return MIToken::lparen;
  case ')':
    return MIToken::rparen;
  case '{':
    return MIToken::lbrace;
  case '}':
    return MIToken::rbrace;
  case '+':
    return MIToken::plus;
  case '-':
    return MIToken::minus;
  case '<':
    return MIToken::less;
  case '>':
    return MIToken::greater;
  default:
    return MIToken::Error;
  }
}

static Cursor maybeLexSymbol(Cursor C, MIToken &Token) {
  MIToken::TokenKind Kind;
  unsigned Length = 1;
  if (C.peek() == ':' && C.peek(1) == ':') {
    Kind = MIToken::coloncolon;
    Length = 2;
  } else
    Kind = symbolToken(C.peek());
  if (Kind == MIToken::Error)
    return None;
  auto Range = C;
  C.advance(Length);
  Token.reset(Kind, Range.upto(C));
  return C;
}

static Cursor maybeLexNewline(Cursor C, MIToken &Token) {
  if (!isNewlineChar(C.peek()))
    return None;
  auto Range = C;
  C.advance();
  Token.reset(MIToken::Newline, Range.upto(C));
  return C;
}

static Cursor maybeLexEscapedIRValue(Cursor C, MIToken &Token,
                                     ErrorCallbackType ErrorCallback) {
  if (C.peek() != '`')
    return None;
  auto Range = C;
  C.advance();
  auto StrRange = C;
  while (C.peek() != '`') {
    if (C.isEOF() || isNewlineChar(C.peek())) {
      ErrorCallback(
          C.location(),
          "end of machine instruction reached before the closing '`'");
      Token.reset(MIToken::Error, Range.remaining());
      return C;
    }
    C.advance();
  }
  StringRef Value = StrRange.upto(C);
  C.advance();
  Token.reset(MIToken::QuotedIRValue, Range.upto(C)).setStringValue(Value);
  return C;
}

StringRef llvm::lexMIToken(StringRef Source, MIToken &Token,
                           ErrorCallbackType ErrorCallback) {
  auto C = skipComment(skipWhitespace(Cursor(Source)));
  if (C.isEOF()) {
    Token.reset(MIToken::Eof, C.remaining());
    return C.remaining();
  }

  if (Cursor R = maybeLexIntegerOrScalarType(C, Token))
    return R.remaining();
  if (Cursor R = maybeLexMachineBasicBlock(C, Token, ErrorCallback))
    return R.remaining();
  if (Cursor R = maybeLexIdentifier(C, Token))
    return R.remaining();
  if (Cursor R = maybeLexJumpTableIndex(C, Token))
    return R.remaining();
  if (Cursor R = maybeLexStackObject(C, Token))
    return R.remaining();
  if (Cursor R = maybeLexFixedStackObject(C, Token))
    return R.remaining();
  if (Cursor R = maybeLexConstantPoolItem(C, Token))
    return R.remaining();
  if (Cursor R = maybeLexSubRegisterIndex(C, Token, ErrorCallback))
    return R.remaining();
  if (Cursor R = maybeLexIRBlock(C, Token, ErrorCallback))
    return R.remaining();
  if (Cursor R = maybeLexIRValue(C, Token, ErrorCallback))
    return R.remaining();
  if (Cursor R = maybeLexRegister(C, Token))
    return R.remaining();
  if (Cursor R = maybeLexGlobalValue(C, Token, ErrorCallback))
    return R.remaining();
  if (Cursor R = maybeLexExternalSymbol(C, Token, ErrorCallback))
    return R.remaining();
  if (Cursor R = maybeLexHexFloatingPointLiteral(C, Token))
    return R.remaining();
  if (Cursor R = maybeLexNumericalLiteral(C, Token))
    return R.remaining();
  if (Cursor R = maybeLexExlaim(C, Token, ErrorCallback))
    return R.remaining();
  if (Cursor R = maybeLexSymbol(C, Token))
    return R.remaining();
  if (Cursor R = maybeLexNewline(C, Token))
    return R.remaining();
  if (Cursor R = maybeLexEscapedIRValue(C, Token, ErrorCallback))
    return R.remaining();

  Token.reset(MIToken::Error, C.remaining());
  ErrorCallback(C.location(),
                Twine("unexpected character '") + Twine(C.peek()) + "'");
  return C.remaining();
}
