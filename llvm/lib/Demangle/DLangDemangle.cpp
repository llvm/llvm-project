//===--- DLangDemangle.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines a demangler for the D programming language as specified
/// in the ABI specification, available at:
/// https://dlang.org/spec/abi.html#name_mangling
///
//===----------------------------------------------------------------------===//

#include "llvm/Demangle/Demangle.h"
#include "llvm/Demangle/StringViewExtras.h"
#include "llvm/Demangle/Utility.h"

#include <cctype>
#include <cstring>
#include <limits>
#include <string_view>

using namespace llvm;
using llvm::itanium_demangle::OutputBuffer;
using llvm::itanium_demangle::starts_with;

namespace {

/// Demangle information structure.
struct Demangler {
  /// Initialize the information structure we use to pass around information.
  ///
  /// \param Mangled String to demangle.
  Demangler(std::string_view Mangled);

  /// Extract and demangle the mangled symbol and append it to the output
  /// string.
  ///
  /// \param Demangled Output buffer to write the demangled name.
  ///
  /// \return The remaining string on success or nullptr on failure.
  ///
  /// \see https://dlang.org/spec/abi.html#name_mangling .
  /// \see https://dlang.org/spec/abi.html#MangledName .
  const char *parseMangle(OutputBuffer *Demangled);

private:
  /// Extract and demangle a given mangled symbol and append it to the output
  /// string.
  ///
  /// \param Demangled output buffer to write the demangled name.
  /// \param Mangled mangled symbol to be demangled.
  /// \param PrintType writes the type information of the symbol to the
  /// demangled name.
  ///
  /// \see https://dlang.org/spec/abi.html#name_mangling .
  /// \see https://dlang.org/spec/abi.html#MangledName .
  void parseMangle(OutputBuffer *Demangled, std::string_view &Mangled,
                   bool PrintType = true);

  /// Extract the number from a given string.
  ///
  /// \param Mangled string to extract the number.
  /// \param Ret assigned result value.
  ///
  /// \note Ret larger than UINT_MAX is considered a failure.
  ///
  /// \see https://dlang.org/spec/abi.html#Number .
  void decodeNumber(std::string_view &Mangled, unsigned long &Ret);

  /// Extract the back reference position from a given string.
  ///
  /// \param Mangled string to extract the back reference position.
  /// \param Ret assigned result value.
  ///
  /// \return true on success, false on error.
  ///
  /// \note Ret is always >= 0 on success, and unspecified on failure
  ///
  /// \see https://dlang.org/spec/abi.html#back_ref .
  /// \see https://dlang.org/spec/abi.html#NumberBackRef .
  bool decodeBackrefPos(std::string_view &Mangled, long &Ret);

  /// Extract the symbol pointed by the back reference form a given string.
  ///
  /// \param Mangled string to extract the back reference position.
  /// \param Ret assigned result value.
  ///
  /// \return true on success, false on error.
  ///
  /// \see https://dlang.org/spec/abi.html#back_ref .
  bool decodeBackref(std::string_view &Mangled, std::string_view &Ret);

  /// Extract and demangle backreferenced symbol from a given mangled symbol
  /// and append it to the output string.
  ///
  /// \param Demangled output buffer to write the demangled name.
  /// \param Mangled mangled symbol to be demangled.
  ///
  /// \see https://dlang.org/spec/abi.html#back_ref .
  /// \see https://dlang.org/spec/abi.html#IdentifierBackRef .
  void parseSymbolBackref(OutputBuffer *Demangled, std::string_view &Mangled);

  /// Extract and demangle backreferenced type from a given mangled symbol
  /// and append it to the output string.
  ///
  /// \param Demangled output buffer to write the demangled name.
  /// \param Mangled mangled symbol to be demangled.
  ///
  /// \see https://dlang.org/spec/abi.html#back_ref .
  /// \see https://dlang.org/spec/abi.html#TypeBackRef .
  void parseTypeBackref(OutputBuffer *Demangled, std::string_view &Mangled);

  /// Check whether it is the beginning of a symbol name.
  ///
  /// \param Mangled string to extract the symbol name.
  ///
  /// \return true on success, false otherwise.
  ///
  /// \see https://dlang.org/spec/abi.html#SymbolName .
  bool isSymbolName(std::string_view Mangled);

  /// Extract and demangle an identifier from a given mangled symbol append it
  /// to the output string.
  ///
  /// \param Demangled Output buffer to write the demangled name.
  /// \param Mangled Mangled symbol to be demangled.
  ///
  /// \see https://dlang.org/spec/abi.html#SymbolName .
  void parseIdentifier(OutputBuffer *Demangled, std::string_view &Mangled);

  /// Extract and demangle the plain identifier from a given mangled symbol and
  /// prepend/append it to the output string, with a special treatment for some
  /// magic compiler generated symbols.
  ///
  /// \param Demangled Output buffer to write the demangled name.
  /// \param Mangled Mangled symbol to be demangled.
  /// \param Len Length of the mangled symbol name.
  ///
  /// \see https://dlang.org/spec/abi.html#LName .
  void parseLName(OutputBuffer *Demangled, std::string_view &Mangled,
                  unsigned long Len);

  /// Extract and demangle the qualified symbol from a given mangled symbol
  /// append it to the output string.
  ///
  /// \param Demangled Output buffer to write the demangled name.
  /// \param Mangled Mangled symbol to be demangled.
  ///
  /// \see https://dlang.org/spec/abi.html#QualifiedName .
  void parseQualified(OutputBuffer *Demangled, std::string_view &Mangled);

  /// Extract and demangle a type from a given mangled symbol append it to
  /// the output string.
  ///
  /// \param Demangled Output buffer to write the demangled name.
  /// \param Mangled mangled symbol to be demangled.
  ///
  /// \return true on success, false on error.
  ///
  /// \see https://dlang.org/spec/abi.html#Type .
  bool parseType(OutputBuffer *Demangled, std::string_view &Mangled);

  /// Extract and demangle a function type from a given mangled symbol append it
  /// to the output string.
  ///
  /// \param Demangled Output buffer to write the demangled name.
  /// \param Mangled mangled symbol to be demangled.
  /// \param IsDelegate Flag to determine if an function or a delegate is
  /// currently being demangled.
  ///
  /// \return true on success, false on error.
  ///
  /// \see https://dlang.org/spec/abi.html#TypeFunction .
  bool parseTypeFunction(OutputBuffer *Demangled, std::string_view &Mangled,
                         bool IsDelegate);

  /// Extract and demangle a calling convention from a given mangled symbol
  /// append it to the output string.
  ///
  /// \param Demangled Output buffer to write the demangled name.
  /// \param Mangled mangled symbol to be demangled.
  ///
  /// \return true on success, false on error.
  ///
  /// \see https://dlang.org/spec/abi.html#CallConvention
  bool parseCallConvention(OutputBuffer *Demangled, std::string_view &Mangled);

  /// Extract and demangle function parameters from a given mangled symbol
  /// append it to the output string.
  ///
  /// \param Demangled Output buffer to write the demangled name.
  /// \param Mangled mangled symbol to be demangled.
  ///
  /// \return true on success, false on error.
  ///
  /// \see https://dlang.org/spec/abi.html#Parameters
  bool parseFuncParameters(OutputBuffer *Demangled, std::string_view &Mangled);

  /// Extract function attributes from a given mangled symbol.
  ///
  /// \param Mangled mangled symbol to be demangled.
  ///
  /// \return bitmap where each bit represents one function attribute.
  ///
  /// \see https://dlang.org/spec/abi.html#FuncAttrs
  unsigned short parseFuncAttributes(std::string_view &Mangled);

  /// Demangle function attributes append it to the output string.
  ///
  /// \param Demangled Output buffer to write the demangled name.
  /// \param Attributes The extracted function attributes.
  ///
  /// \see https://dlang.org/spec/abi.html#FuncAttrs
  void demangleFuncAttributes(OutputBuffer *Demangled,
                              unsigned short Attributes);

  /// Extract modifiers from a given mangled symbol.
  ///
  /// \param Mangled mangled symbol to be demangled.
  ///
  /// \return bitmap where each bit represents one function attribute.
  ///
  /// \see https://dlang.org/spec/abi.html#TypeModifiers
  unsigned short parseModifiers(std::string_view &Mangled);

  /// Demangle modifiers append it to the output string.
  ///
  /// \param Demangled Output buffer to write the demangled name.
  /// \param Modifiers The extracted modifiers.
  ///
  /// \see https://dlang.org/spec/abi.html#TypeModifiers
  void demangleModifiers(OutputBuffer *Demangled, unsigned short Modifiers);

  /// Extract and demangle a function type without a return type from a given
  /// mangled symbol append it to the output string.
  ///
  /// \param Attrs Output buffer to write the demangled attributes.
  /// \param Demangled Output buffer to write the demangled name.
  /// \param Mangled mangled symbol to be demangled.
  ///
  /// \return true on success, false on error.
  ///
  /// \see https://dlang.org/spec/abi.html#TypeFunctionNoReturn
  bool parseFunctionTypeNoReturn(OutputBuffer *Attrs, OutputBuffer *Demangled,
                                 std::string_view &Mangled);

  /// Extract and demangle a template instance from a given mangled symbol
  /// append it to the output string.
  ///
  /// \param Demangled Output buffer to write the demangled name.
  /// \param Mangled mangled symbol to be demangled.
  ///
  /// \see https://dlang.org/spec/abi.html#TemplateInstanceName
  void parseTemplateInstanceName(OutputBuffer *Demangled,
                                 std::string_view &Mangled);

  /// Extract and demangle a template argument value from a given mangled symbol
  /// append it to the output string.
  ///
  /// \param Demangled Output buffer to write the demangled name.
  /// \param Mangled mangled symbol to be demangled.
  /// \param Type mangled template argument type.
  /// \param DemangledType Output buffer containing the already
  /// demangled template argument type if present.
  ///
  /// \see https://dlang.org/spec/abi.html#Value
  void parseValue(OutputBuffer *Demangled, std::string_view &Mangled,
                  char Type = '\0', OutputBuffer *DemangledType = nullptr);

  /// Writes an hexadecimal number to the output string.
  ///
  /// \param Demangled Output buffer to write the hexadecimal number.
  /// \param Val the numeric value to write.
  /// \param Width the width of the hexadecimal number.
  void printHexNumber(OutputBuffer *Demangled, unsigned long Val,
                      unsigned Width = 0);

  /// Extract and demangle a numeric template argument value from a given
  /// mangled symbol append it to the output string.
  ///
  /// \param Demangled Output buffer to write the demangled name.
  /// \param Mangled mangled symbol to be demangled.
  /// \param Type mangled template argument type.
  void parseIntegerValue(OutputBuffer *Demangled, std::string_view &Mangled,
                         char Type = '\0');

  /// Extract and demangle a numeric template argument value from a given
  /// mangled symbol append it to the output string.
  ///
  /// \param Demangled Output buffer to write the demangled name.
  /// \param Mangled mangled symbol to be demangled.
  ///
  /// \see https://dlang.org/spec/abi.html#HexFloat
  void parseRealValue(OutputBuffer *Demangled, std::string_view &Mangled);

  /// An immutable view of the string we are demangling.
  const std::string_view Str;
  /// The index of the last back reference.
  int LastBackref;
};

} // namespace

inline bool isHexDigit(char Val) {
  return std::isdigit(Val) || (Val >= 'A' && Val <= 'F');
}

void Demangler::decodeNumber(std::string_view &Mangled, unsigned long &Ret) {
  // Clear Mangled if trying to extract something that isn't a digit.
  if (Mangled.empty()) {
    Mangled = {};
    return;
  }

  if (!std::isdigit(Mangled.front())) {
    Mangled = {};
    return;
  }

  unsigned long Val = 0;

  do {
    unsigned long Digit = Mangled[0] - '0';

    // Check for overflow.
    if (Val > (std::numeric_limits<unsigned int>::max() - Digit) / 10) {
      Mangled = {};
      return;
    }

    Val = Val * 10 + Digit;
    Mangled.remove_prefix(1);
  } while (!Mangled.empty() && std::isdigit(Mangled.front()));

  if (Mangled.empty()) {
    Mangled = {};
    return;
  }

  Ret = Val;
}

bool Demangler::decodeBackrefPos(std::string_view &Mangled, long &Ret) {
  // Return nullptr if trying to extract something that isn't a digit
  if (Mangled.empty()) {
    Mangled = {};
    return false;
  }
  // Any identifier or non-basic type that has been emitted to the mangled
  // symbol before will not be emitted again, but is referenced by a special
  // sequence encoding the relative position of the original occurrence in the
  // mangled symbol name.
  // Numbers in back references are encoded with base 26 by upper case letters
  // A-Z for higher digits but lower case letters a-z for the last digit.
  //    NumberBackRef:
  //        [a-z]
  //        [A-Z] NumberBackRef
  //        ^
  unsigned long Val = 0;

  while (!Mangled.empty() && std::isalpha(Mangled.front())) {
    // Check for overflow
    if (Val > (std::numeric_limits<unsigned long>::max() - 25) / 26)
      break;

    Val *= 26;

    if (Mangled[0] >= 'a' && Mangled[0] <= 'z') {
      Val += Mangled[0] - 'a';
      if ((long)Val <= 0)
        break;
      Ret = Val;
      Mangled.remove_prefix(1);
      return true;
    }

    Val += Mangled[0] - 'A';
    Mangled.remove_prefix(1);
  }

  Mangled = {};
  return false;
}

bool Demangler::decodeBackref(std::string_view &Mangled,
                              std::string_view &Ret) {
  assert(!Mangled.empty() && Mangled.front() == 'Q' &&
         "Invalid back reference!");
  Ret = {};

  // Position of 'Q'
  const char *Qpos = Mangled.data();
  long RefPos;
  Mangled.remove_prefix(1);

  if (!decodeBackrefPos(Mangled, RefPos)) {
    Mangled = {};
    return false;
  }

  if (RefPos > Qpos - Str.data()) {
    Mangled = {};
    return false;
  }

  // Set the position of the back reference.
  Ret = Qpos - RefPos;

  return true;
}

void Demangler::parseSymbolBackref(OutputBuffer *Demangled,
                                   std::string_view &Mangled) {
  // An identifier back reference always points to a digit 0 to 9.
  //    IdentifierBackRef:
  //        Q NumberBackRef
  //        ^
  unsigned long Len;

  // Get position of the back reference
  std::string_view Backref;
  if (!decodeBackref(Mangled, Backref)) {
    Mangled = {};
    return;
  }

  // Must point to a simple identifier
  decodeNumber(Backref, Len);
  if (Backref.empty() || Backref.length() < Len) {
    Mangled = {};
    return;
  }

  parseLName(Demangled, Backref, Len);
  if (Backref.empty())
    Mangled = {};
}

void Demangler::parseTypeBackref(OutputBuffer *Demangled,
                                 std::string_view &Mangled) {
  // A type back reference always points to a letter.
  //    TypeBackRef:
  //        Q NumberBackRef
  //        ^

  // If we appear to be moving backwards through the mangle string, then
  // bail as this may be a recursive back reference.
  if (Mangled.data() - Str.data() >= LastBackref) {
    Mangled = {};
    return;
  }

  int SaveRefPos = LastBackref;
  LastBackref = Mangled.data() - Str.data();

  // Get position of the back reference.
  std::string_view Backref;
  if (!decodeBackref(Mangled, Backref)) {
    Mangled = {};
    return;
  }

  // Can't decode back reference.
  if (Backref.empty()) {
    Mangled = {};
    return;
  }

  // TODO: Add support for function type back references.
  if (!parseType(Demangled, Backref))
    Mangled = {};

  LastBackref = SaveRefPos;

  if (Backref.empty())
    Mangled = {};
}

bool Demangler::isSymbolName(std::string_view Mangled) {
  long Ret;
  const char *Qref = Mangled.data();

  if (std::isdigit(Mangled.front()))
    return true;

  if (Mangled.front() == '_')
    return true;

  if (Mangled.front() != 'Q')
    return false;

  Mangled.remove_prefix(1);
  bool Valid = decodeBackrefPos(Mangled, Ret);
  if (!Valid || Ret > Qref - Str.data())
    return false;

  return std::isdigit(Qref[-Ret]);
}

void Demangler::parseMangle(OutputBuffer *Demangled, std::string_view &Mangled,
                            bool PrintType) {
  // A D mangled symbol is comprised of both scope and type information.
  //    MangleName:
  //        _D QualifiedName Type
  //        _D QualifiedName Z
  //        ^
  // The caller should have guaranteed that the start pointer is at the
  // above location.
  // Note that type is never a function type, but only the return type of
  // a function or the type of a variable.
  Mangled.remove_prefix(2);

  size_t NotFirst = false;
  OutputBuffer Attrs;
  do {
    // Skip over anonymous symbols.
    if (!Mangled.empty() && Mangled.front() == '0') {
      do
        Mangled.remove_prefix(1);
      while (!Mangled.empty() && Mangled.front() == '0');

      continue;
    }

    // Ignore all attributes for parent symbols
    Attrs.setCurrentPosition(0);

    if (NotFirst)
      *Demangled << '.';
    NotFirst = true;

    parseIdentifier(Demangled, Mangled);
    parseFunctionTypeNoReturn(&Attrs, Demangled, Mangled);

  } while (!Mangled.empty() && isSymbolName(Mangled));

  size_t TypePos = 0;
  if (!Attrs.empty()) {
    Attrs << ' ';
    Demangled->insert(0, Attrs.getBuffer(), Attrs.getCurrentPosition());
    TypePos = Attrs.getCurrentPosition();
  }
  std::free(Attrs.getBuffer());

  if (Mangled.empty()) {
    Mangled = {};
    return;
  }

  // Artificial symbols end with 'Z' and have no type.
  if (Mangled.front() == 'Z') {
    Mangled.remove_prefix(1);
  } else {
    OutputBuffer tmp;
    if (!parseType(&tmp, Mangled))
      Mangled = {};
    else if (PrintType) {
      tmp << ' ';
      Demangled->insert(TypePos, tmp.getBuffer(), tmp.getCurrentPosition());
    }
    std::free(tmp.getBuffer());
  }
}

void Demangler::parseQualified(OutputBuffer *Demangled,
                               std::string_view &Mangled) {
  // Qualified names are identifiers separated by their encoded length.
  // Nested functions also encode their argument types without specifying
  // what they return.
  //    QualifiedName:
  //        SymbolFunctionName
  //        SymbolFunctionName QualifiedName
  //        ^
  //    SymbolFunctionName:
  //        SymbolName
  //        SymbolName TypeFunctionNoReturn
  //        SymbolName M TypeFunctionNoReturn
  //        SymbolName M TypeModifiers TypeFunctionNoReturn
  // The start pointer should be at the above location.

  // Whether it has more than one symbol
  size_t NotFirst = false;
  do {
    // Skip over anonymous symbols.
    if (!Mangled.empty() && Mangled.front() == '0') {
      do
        Mangled.remove_prefix(1);
      while (!Mangled.empty() && Mangled.front() == '0');

      continue;
    }

    if (NotFirst)
      *Demangled << '.';
    NotFirst = true;

    parseIdentifier(Demangled, Mangled);
    parseFunctionTypeNoReturn(Demangled, Demangled, Mangled);

  } while (!Mangled.empty() && isSymbolName(Mangled));
}

void Demangler::parseIdentifier(OutputBuffer *Demangled,
                                std::string_view &Mangled) {
  if (Mangled.empty()) {
    Mangled = {};
    return;
  }

  if (Mangled.front() == 'Q')
    return parseSymbolBackref(Demangled, Mangled);

  if (Mangled.front() == '_')
    return parseTemplateInstanceName(Demangled, Mangled);

  unsigned long Len;
  decodeNumber(Mangled, Len);

  if (Mangled.empty()) {
    Mangled = {};
    return;
  }
  if (!Len || Mangled.length() < Len) {
    Mangled = {};
    return;
  }

  // TODO: Parse template instances with a length prefix.

  // There can be multiple different declarations in the same function that
  // have the same mangled name.  To make the mangled names unique, a fake
  // parent in the form `__Sddd' is added to the symbol.
  if (Len >= 4 && starts_with(Mangled, "__S")) {
    const size_t SuffixLen = Mangled.length() - Len;
    std::string_view P = Mangled.substr(3);
    while (P.length() > SuffixLen && std::isdigit(P.front()))
      P.remove_prefix(1);
    if (P.length() == SuffixLen) {
      // Skip over the fake parent.
      Mangled.remove_prefix(Len);
      return parseIdentifier(Demangled, Mangled);
    }

    // Else demangle it as a plain identifier.
  }

  parseLName(Demangled, Mangled, Len);
}

bool Demangler::parseType(OutputBuffer *Demangled, std::string_view &Mangled) {
  if (Mangled.empty()) {
    Mangled = {};
    return false;
  }

  switch (Mangled.front()) {
  // Type qualifiers.
  case 'O': {
    *Demangled << "shared(";
    Mangled.remove_prefix(1);
    if (!parseType(Demangled, Mangled)) {
      Mangled = {};
      return false;
    }
    *Demangled << ')';
    return true;
  }

  case 'x': {
    *Demangled << "const(";
    Mangled.remove_prefix(1);
    if (!parseType(Demangled, Mangled)) {
      Mangled = {};
      return false;
    }
    *Demangled << ')';
    return true;
  }

  case 'y': {
    *Demangled << "immutable(";
    Mangled.remove_prefix(1);
    if (!parseType(Demangled, Mangled)) {
      Mangled = {};
      return false;
    }
    *Demangled << ')';
    return true;
  }

  // Function types.
  case 'F':
  case 'U':
  case 'W':
  case 'V':
  case 'R':
    return parseTypeFunction(Demangled, Mangled, false);

  // Array types.
  case 'A': {
    Mangled.remove_prefix(1);
    if (!parseType(Demangled, Mangled)) {
      Mangled = {};
      return false;
    }
    *Demangled << "[]";
    return true;
  }

  case 'G': {
    Mangled.remove_prefix(1);
    unsigned long len;
    decodeNumber(Mangled, len);
    if (!parseType(Demangled, Mangled)) {
      Mangled = {};
      return false;
    }
    *Demangled << '[' << len << ']';
    return true;
  }

  case 'H': {
    Mangled.remove_prefix(1);
    OutputBuffer tmp;
    if (!parseType(&tmp, Mangled)) {
      std::free(tmp.getBuffer());
      Mangled = {};
      return false;
    }
    if (!parseType(Demangled, Mangled)) {
      std::free(tmp.getBuffer());
      Mangled = {};
      return false;
    }
    *Demangled << '[';
    *Demangled << tmp.getBuffer();
    *Demangled << ']';
    std::free(tmp.getBuffer());
    return true;
  }

  case 'P': {
    Mangled.remove_prefix(1);
    if (!parseType(Demangled, Mangled)) {
      Mangled = {};
      return false;
    }
    *Demangled << "*";
    return true;
  }

  // Delegate types.
  case 'D': {
    Mangled.remove_prefix(1);
    auto Modifiers = parseModifiers(Mangled);
    if (!parseTypeFunction(Demangled, Mangled, true)) {
      Mangled = {};
      return false;
    }
    if (Modifiers > 0) {
      *Demangled << ' ';
      demangleModifiers(Demangled, Modifiers);
    }
    return true;
  }

  // Compound types.
  case 'I':
  case 'C':
  case 'S':
  case 'E':
  case 'T': {
    Mangled.remove_prefix(1);
    parseQualified(Demangled, Mangled);
    return true;
  }

  // TODO: Parse tuple types.

  // Cent types.
  case 'z': {
    Mangled.remove_prefix(1);
    switch (Mangled.front()) {
    case 'i':
      *Demangled << "cent";
      Mangled.remove_prefix(1);
      return true;
    case 'k':
      *Demangled << "ucent";
      Mangled.remove_prefix(1);
      return true;
    default:
      Mangled = {};
      return false;
    }
  }

  // Misc.
  case 'N': {
    Mangled.remove_prefix(1);
    switch (Mangled.front()) {
    case 'n':
      *Demangled << "noreturn";
      Mangled.remove_prefix(1);
      return true;
    case 'h':
      *Demangled << "__vector(";
      Mangled.remove_prefix(1);
      if (!parseType(Demangled, Mangled)) {
        Mangled = {};
        return false;
      }
      *Demangled << ')';
      return true;
    case 'g':
      *Demangled << "inout(";
      Mangled.remove_prefix(1);
      if (!parseType(Demangled, Mangled)) {
        Mangled = {};
        return false;
      }
      *Demangled << ')';
      return true;
    }
    Mangled = {};
    return false;
  }

  // Back referenced type.
  case 'Q': {
    parseTypeBackref(Demangled, Mangled);
    return true;
  }

  default:
    char c = Mangled.front();
    // Basic types.
    if (c >= 'a' && c <= 'w') {
      Mangled.remove_prefix(1);

      static const char *Primitives[] = {
          "char",    // a
          "bool",    // b
          "creal",   // c
          "double",  // d
          "real",    // e
          "float",   // f
          "byte",    // g
          "ubyte",   // h
          "int",     // i
          "ireal",   // j
          "uint",    // k
          "long",    // l
          "ulong",   // m
          0,         // n
          "ifloat",  // o
          "idouble", // p
          "cfloat",  // q
          "cdouble", // r
          "short",   // s
          "ushort",  // t
          "wchar",   // u
          "void",    // v
          "dchar",   // w
      };

      *Demangled << Primitives[c - 'a'];
      return true;
    }

    // unhandled.
    Mangled = {};
    return false;
  }
}

bool Demangler::parseTypeFunction(OutputBuffer *Demangled,
                                  std::string_view &Mangled, bool IsDelegate) {
  if (!parseCallConvention(Demangled, Mangled)) {
    Mangled = {};
    return false;
  }

  auto funcAttrs = parseFuncAttributes(Mangled);
  if (Mangled.empty()) {
    Mangled = {};
    return false;
  }

  auto begin = Demangled->getCurrentPosition();

  if (IsDelegate)
    *Demangled << "delegate";
  else
    *Demangled << "function";

  *Demangled << '(';
  if (!parseFuncParameters(Demangled, Mangled)) {
    Mangled = {};
    return false;
  }
  *Demangled << ')';

  OutputBuffer tmp;
  if (!parseType(&tmp, Mangled)) {
    std::free(tmp.getBuffer());
    Mangled = {};
    return false;
  }
  tmp << ' ';
  Demangled->insert(begin, tmp.getBuffer(), tmp.getCurrentPosition());
  std::free(tmp.getBuffer());

  if (funcAttrs > 0) {
    *Demangled << ' ';
    demangleFuncAttributes(Demangled, funcAttrs);
  }

  return true;
}

bool Demangler::parseCallConvention(OutputBuffer *Demangled,
                                    std::string_view &Mangled) {
  switch (Mangled.front()) {
  case 'F':
    Mangled.remove_prefix(1);
    return true;
  case 'U':
    *Demangled << "extern (C) ";
    Mangled.remove_prefix(1);
    return true;
  case 'W':
    *Demangled << "extern (Windows) ";
    Mangled.remove_prefix(1);
    return true;
  case 'R':
    *Demangled << "extern (C++) ";
    Mangled.remove_prefix(1);
    return true;
  default:
    Mangled = {};
    return false;
  }
}

bool Demangler::parseFuncParameters(OutputBuffer *Demangled,
                                    std::string_view &Mangled) {
  for (size_t i = 0; true; i++) {
    if (Mangled.empty()) {
      Mangled = {};
      return false;
    }

    switch (Mangled.front()) {
    case 'X':
      *Demangled << "...";
      Mangled.remove_prefix(1);
      return true;
    case 'Y':
      *Demangled << ", ...";
      Mangled.remove_prefix(1);
      return true;
    case 'Z':
      Mangled.remove_prefix(1);
      return true;
    }

    if (i)
      *Demangled << ", ";

    while (1) {
      if (Mangled.empty()) {
        Mangled = {};
        return false;
      }

      switch (Mangled.front()) {
      case 'M':
        *Demangled << "scope ";
        Mangled.remove_prefix(1);
        continue;
      case 'N':
        Mangled.remove_prefix(1);
        if (Mangled.front() == 'k') {
          *Demangled << "return ";
          Mangled.remove_prefix(1);
          continue;
        } else
          goto afterAttrLoop;
      default:
        goto afterAttrLoop;
      }
    }
  afterAttrLoop:

    switch (Mangled.front()) {
    case 'I': {
      *Demangled << "in ";
      Mangled.remove_prefix(1);
      if (Mangled.front() == 'K')
        goto refStorageClass;
      if (!parseType(Demangled, Mangled)) {
        Mangled = {};
        return false;
      }
      continue;
    }

    case 'K':
    refStorageClass: {
      *Demangled << "ref ";
      Mangled.remove_prefix(1);
      if (!parseType(Demangled, Mangled)) {
        Mangled = {};
        return false;
      }
      continue;
    }

    case 'J': {
      *Demangled << "out ";
      Mangled.remove_prefix(1);
      if (!parseType(Demangled, Mangled)) {
        Mangled = {};
        return false;
      }
      continue;
    }

    case 'L': {
      *Demangled << "lazy ";
      Mangled.remove_prefix(1);
      if (!parseType(Demangled, Mangled)) {
        Mangled = {};
        return false;
      }
      continue;
    }

    default:
      if (!parseType(Demangled, Mangled)) {
        Mangled = {};
        return false;
      }
    }
  }
}

enum FunctionAttribute {
  Pure = (1 << 0),
  NoThrow = (1 << 1),
  Ref = (1 << 2),
  Property = (1 << 3),
  Trusted = (1 << 4),
  Safe = (1 << 5),
  NoGC = (1 << 6),
  Return = (1 << 7),
  Scope = (1 << 8),
  Live = (1 << 9),
  ReturnScope = (1 << 10),
  ScopeReturn = (1 << 11),
};
#define FUNCTIONATTR_MAX 12

unsigned short Demangler::parseFuncAttributes(std::string_view &Mangled) {
  if (Mangled.empty()) {
    return 0;
  }

  unsigned short Result = 0;
  while (Mangled.front() == 'N') {
    if (Mangled.length() < 2) {
      return Result;
    }

    switch (Mangled.at(1)) {
    case 'a':
      Result |= FunctionAttribute::Pure;
      Mangled.remove_prefix(2);
      continue;
    case 'b':
      Result |= FunctionAttribute::NoThrow;
      Mangled.remove_prefix(2);
      continue;
    case 'c':
      Result |= FunctionAttribute::Ref;
      Mangled.remove_prefix(2);
      continue;
    case 'd':
      Result |= FunctionAttribute::Property;
      Mangled.remove_prefix(2);
      continue;
    case 'e':
      Result |= FunctionAttribute::Trusted;
      Mangled.remove_prefix(2);
      continue;
    case 'f':
      Result |= FunctionAttribute::Safe;
      Mangled.remove_prefix(2);
      continue;
    case 'i':
      Result |= FunctionAttribute::NoGC;
      Mangled.remove_prefix(2);
      continue;
    case 'j': {
      Mangled.remove_prefix(1);
      if (starts_with(Mangled, "Nl")) {
        Result |= FunctionAttribute::ReturnScope;
        Mangled.remove_prefix(2);
      } else
        Result |= FunctionAttribute::Return;
      continue;
    }
    case 'l': {
      Mangled.remove_prefix(1);
      if (starts_with(Mangled, "Nj")) {
        Result |= FunctionAttribute::ScopeReturn;
        Mangled.remove_prefix(2);
      } else
        Result |= FunctionAttribute::Scope;
      continue;
    }
    case 'm':
      Result |= FunctionAttribute::Live;
      Mangled.remove_prefix(2);
      continue;
    default:
      return Result;
    }
  }
  return Result;
}

void Demangler::demangleFuncAttributes(OutputBuffer *Demangled,
                                       unsigned short Attributes) {
  static const char *AttributeNames[] = {
      "pure",  "nothrow", "ref",   "@property", "@trusted",     "@safe",
      "@nogc", "return",  "scope", "@live",     "return scope", "scope return",
  };

  bool NeedSpace = false;
  for (unsigned short i = 0; i < FUNCTIONATTR_MAX; i++) {
    if (Attributes & (1 << i)) {
      if (NeedSpace)
        *Demangled << ' ';
      *Demangled << AttributeNames[i];
      NeedSpace = true;
    }
  }
}

enum Modifier {
  Const = (1 << 0),
  Immutable = (1 << 1),
  Shared = (1 << 2),
  InOut = (1 << 3),
};
#define MODIFIER_MAX 4

unsigned short Demangler::parseModifiers(std::string_view &Mangled) {
  unsigned short Result = 0;
  switch (Mangled.front()) {
  case 'y':
    Mangled.remove_prefix(1);
    return Modifier::Immutable;
  case 'O':
    Mangled.remove_prefix(1);
    Result |= Modifier::Shared;
    switch (Mangled.front()) {
    case 'x':
      goto constMod;
    case 'N':
      goto wildMod;
    default:
      return Modifier::Shared;
    }
  case 'N':
  wildMod:
    if (Mangled.size() > 1 && Mangled.at(1) != 'g')
      return Result;
    Mangled.remove_prefix(2);
    Result |= Modifier::InOut;
    if (!Mangled.empty() && Mangled.front() == 'x')
      goto constMod;
    return Result;
  case 'x':
  constMod:
    Mangled.remove_prefix(1);
    Result |= Modifier::Const;
    return Result;
  default:
    return 0;
  }
}

void Demangler::demangleModifiers(OutputBuffer *Demangled,
                                  unsigned short Modifiers) {
  static const char *ModifierNames[] = {
      "const",
      "immutable",
      "shared",
      "inout",
  };

  bool NeedSpace = false;
  for (unsigned short i = 0; i < MODIFIER_MAX; i++) {
    if (Modifiers & (1 << i)) {
      if (NeedSpace)
        *Demangled << ' ';
      *Demangled << ModifierNames[i];
      NeedSpace = true;
    }
  }
}

bool Demangler::parseFunctionTypeNoReturn(OutputBuffer *Attrs,
                                          OutputBuffer *Demangled,
                                          std::string_view &Mangled) {
  if (Mangled.empty())
    return true;

  if (Mangled.front() == 'M') {
    Mangled.remove_prefix(1);
    auto Modifiers = parseModifiers(Mangled);
    if (Modifiers > 0) {
      demangleModifiers(Attrs, Modifiers);
      *Attrs << ' ';
    }
  }

  switch (Mangled.front()) {
  case 'F':
  case 'U':
  case 'W':
  case 'R':
    if (!parseCallConvention(Attrs, Mangled)) {
      Mangled = {};
      return false;
    }

    auto funcAttrs = parseFuncAttributes(Mangled);
    if (funcAttrs > 0)
      demangleFuncAttributes(Attrs, funcAttrs);

    *Demangled << '(';
    if (!parseFuncParameters(Demangled, Mangled)) {
      Mangled = {};
      return false;
    }
    *Demangled << ')';

    return true;
  }
  return true;
}

void Demangler::parseTemplateInstanceName(OutputBuffer *Demangled,
                                          std::string_view &Mangled) {
  // TODO: handle template instances with length prefix

  if (Mangled.length() < 3 || Mangled.substr(0, 3) != "__T") {
    Mangled = {};
    return;
  }
  Mangled.remove_prefix(3);

  unsigned long Len;
  decodeNumber(Mangled, Len);

  if (Mangled.empty()) {
    Mangled = {};
    return;
  }
  if (!Len || Mangled.length() < Len) {
    Mangled = {};
    return;
  }

  parseLName(Demangled, Mangled, Len);

  *Demangled << "!(";

  for (size_t n = 0; true; n++) {
    if (Mangled.empty()) {
      Mangled = {};
      return;
    }

    if (Mangled.front() == 'H')
      Mangled.remove_prefix(1);

    switch (Mangled.front()) {
    case 'Z':
      Mangled.remove_prefix(1);
      goto after;

    case 'T': {
      Mangled.remove_prefix(1);
      if (n)
        *Demangled << ", ";
      if (!parseType(Demangled, Mangled)) {
        Mangled = {};
        return;
      }
      continue;
    }

    case 'V': {
      Mangled.remove_prefix(1);
      if (n)
        *Demangled << ", ";

      char TypeChar = Mangled.front();
      OutputBuffer DemangledType;
      if (!parseType(&DemangledType, Mangled)) {
        std::free(DemangledType.getBuffer());
        Mangled = {};
        return;
      }

      parseValue(Demangled, Mangled, TypeChar, &DemangledType);
      std::free(DemangledType.getBuffer());
      continue;
    }

    case 'S': {
      Mangled.remove_prefix(1);
      if (n)
        *Demangled << ", ";

      if (starts_with(Mangled, "_D"))
        parseMangle(Demangled, Mangled, false);
      else
        parseQualified(Demangled, Mangled);

      continue;
    }

    case 'X': {
      Mangled.remove_prefix(1);
      if (n)
        *Demangled << ", ";
      unsigned long Len;
      decodeNumber(Mangled, Len);
      parseLName(Demangled, Mangled, Len);
      continue;
    }

    default:
      Mangled = {};
      return;
    }
  }
after:

  *Demangled << ')';
}

void Demangler::parseValue(OutputBuffer *Demangled, std::string_view &Mangled,
                           char Type, OutputBuffer *DemangledType) {
  if (Mangled.empty()) {
    Mangled = {};
    return;
  }

  switch (Mangled.front()) {
  case 'n':
    Mangled.remove_prefix(1);
    *Demangled << "null";
    return;

  case 'i':
    Mangled.remove_prefix(1);
    if (!isdigit(Mangled.front())) {
      Mangled = {};
      return;
    }
    parseIntegerValue(Demangled, Mangled, Type);
    return;

  case 'N':
    Mangled.remove_prefix(1);
    *Demangled << '-';
    parseIntegerValue(Demangled, Mangled);
    return;

  case 'e':
    Mangled.remove_prefix(1);
    parseRealValue(Demangled, Mangled);
    return;

  case 'c': {
    Mangled.remove_prefix(1);
    parseRealValue(Demangled, Mangled);
    *Demangled << '+';
    if (Mangled.empty() || Mangled.front() != 'c') {
      Mangled = {};
      return;
    }
    Mangled.remove_prefix(1);
    parseRealValue(Demangled, Mangled);
    *Demangled << 'i';
    return;
  }

  case 'a':
  case 'w':
  case 'd': {
    char Kind = Mangled.front();
    Mangled.remove_prefix(1);

    unsigned long Len;
    decodeNumber(Mangled, Len);

    if (Mangled.front() != '_') {
      Mangled = {};
      return;
    }
    Mangled.remove_prefix(1);

    *Demangled << '"';
    for (unsigned long i = 0; i < Len; i++) {
      if (Mangled.length() < 2) {
        Mangled = {};
        return;
      }

      char HexVal[3] = {Mangled.at(0), Mangled.at(1), 0};
      char Char = (char)(std::stoi(HexVal, 0, 16));
      if (' ' <= Char && Char <= '~')
        *Demangled << Char;
      else
        *Demangled << "\\x" << HexVal;

      Mangled.remove_prefix(2);
    }
    *Demangled << '"';

    if (Kind != 'a')
      *Demangled << Kind;

    return;
  }

  case 'A': {
    Mangled.remove_prefix(1);

    unsigned long Len;
    decodeNumber(Mangled, Len);

    *Demangled << '[';
    for (size_t i = 0; i < Len; i++) {
      if (i)
        *Demangled << ", ";

      if (Type == 'H') {
        parseValue(Demangled, Mangled);
        *Demangled << ':';
      }

      parseValue(Demangled, Mangled);
    }
    *Demangled << ']';
    return;
  }

  case 'S': {
    Mangled.remove_prefix(1);

    unsigned long Len;
    decodeNumber(Mangled, Len);

    if (DemangledType)
      Demangled->insert(Demangled->getCurrentPosition(),
                        DemangledType->getBuffer(),
                        DemangledType->getCurrentPosition());

    *Demangled << '(';
    for (size_t i = 0; i < Len; i++) {
      if (i)
        *Demangled << ", ";
      parseValue(Demangled, Mangled);
    }
    *Demangled << ')';
    return;
  }

    // TODO: f MangledName

  default:
    if (isdigit(Mangled.front())) {
      parseIntegerValue(Demangled, Mangled, Type);
      return;
    }

    Mangled = {};
    return;
  }
}

void Demangler::printHexNumber(OutputBuffer *Demangled, unsigned long Val,
                               unsigned Width) {
  if (Val == 0)
    *Demangled << '0';

  static const char Digits[] = "0123456789ABCDEF";

  for (unsigned i = 0; Width ? (i < Width) : Val; ++i) {
    unsigned char Mod = static_cast<unsigned char>(Val) & 15;
    *Demangled << Digits[Mod];
    Val >>= 4;
  }
}

void Demangler::parseIntegerValue(OutputBuffer *Demangled,
                                  std::string_view &Mangled, char Type) {
  unsigned long Val;
  decodeNumber(Mangled, Val);

  switch (Type) {
  case 'a':
  case 'u':
  case 'w': {
    switch (Val) {
    case '\'':
      *Demangled << "'\\''";
      return;
    case '\\':
      *Demangled << "'\\\\'";
      return;
    case '\a':
      *Demangled << "'\\a'";
      return;
    case '\b':
      *Demangled << "'\\b'";
      return;
    case '\f':
      *Demangled << "'\\f'";
      return;
    case '\n':
      *Demangled << "'\\n'";
      return;
    case '\r':
      *Demangled << "'\\r'";
      return;
    case '\t':
      *Demangled << "'\\t'";
      return;
    case '\v':
      *Demangled << "'\\v'";
      return;
    default:
      switch (Type) {
      case 'a':
        if (Val >= 0x20 && Val < 0x7F)
          *Demangled << '\'' << (char)Val << '\'';
        else {
          *Demangled << "'\\x";
          printHexNumber(Demangled, Val, 2);
          *Demangled << '\'';
        }
        return;

      case 'u':
        *Demangled << "'\\u";
        printHexNumber(Demangled, Val, 4);
        *Demangled << '\'';
        return;

      case 'w':
        *Demangled << "'\\U";
        printHexNumber(Demangled, Val, 8);
        *Demangled << '\'';
        return;
      }
    }

    Mangled = {};
    return;
  }

  case 'b':
    *Demangled << (Val ? "true" : "false");
    return;

  case 'h':
  case 't':
  case 'k':
    *Demangled << Val << 'u';
    return;

  case 'l':
    *Demangled << Val << 'L';
    return;

  case 'm':
    *Demangled << Val << "uL";
    return;

  default:
    *Demangled << Val;
    return;
  }
}

void Demangler::parseRealValue(OutputBuffer *Demangled,
                               std::string_view &Mangled) {
  if (starts_with(Mangled, "INF")) {
    *Demangled << "real.infinity";
    Mangled.remove_prefix(3);
    return;
  } else if (Mangled.front() == 'N') {
    Mangled.remove_prefix(1);
    if (starts_with(Mangled, "INF")) {
      *Demangled << "-real.infinity";
      Mangled.remove_prefix(3);
      return;
    }
    if (starts_with(Mangled, "AN")) {
      *Demangled << "real.nan";
      Mangled.remove_prefix(2);
      return;
    }
    *Demangled << '-';
  }

  *Demangled << "0x";

  while (isHexDigit(Mangled.front())) {
    *Demangled << Mangled.front();
    Mangled.remove_prefix(1);
  }

  if (Mangled.front() != 'P') {
    Mangled = {};
    return;
  }
  Mangled.remove_prefix(1);

  if (Mangled.front() == 'N') {
    Mangled.remove_prefix(1);
    *Demangled << '-';
  } else
    *Demangled << '+';

  while (isdigit(Mangled.front())) {
    *Demangled << Mangled.front();
    Mangled.remove_prefix(1);
  }
}

void Demangler::parseLName(OutputBuffer *Demangled, std::string_view &Mangled,
                           unsigned long Len) {
  switch (Len) {
  case 6:
    if (starts_with(Mangled, "__initZ")) {
      // The static initializer for a given symbol.
      Demangled->prepend("initializer for ");
      Demangled->setCurrentPosition(Demangled->getCurrentPosition() - 1);
      Mangled.remove_prefix(Len);
      return;
    }
    if (starts_with(Mangled, "__vtblZ")) {
      // The vtable symbol for a given class.
      Demangled->prepend("vtable for ");
      Demangled->setCurrentPosition(Demangled->getCurrentPosition() - 1);
      Mangled.remove_prefix(Len);
      return;
    }
    break;

  case 7:
    if (starts_with(Mangled, "__ClassZ")) {
      // The classinfo symbol for a given class.
      Demangled->prepend("ClassInfo for ");
      Demangled->setCurrentPosition(Demangled->getCurrentPosition() - 1);
      Mangled.remove_prefix(Len);
      return;
    }
    break;

  case 11:
    if (starts_with(Mangled, "__InterfaceZ")) {
      // The interface symbol for a given class.
      Demangled->prepend("Interface for ");
      Demangled->setCurrentPosition(Demangled->getCurrentPosition() - 1);
      Mangled.remove_prefix(Len);
      return;
    }
    break;

  case 12:
    if (starts_with(Mangled, "__ModuleInfoZ")) {
      // The ModuleInfo symbol for a given module.
      Demangled->prepend("ModuleInfo for ");
      Demangled->setCurrentPosition(Demangled->getCurrentPosition() - 1);
      Mangled.remove_prefix(Len);
      return;
    }
    break;
  }

  *Demangled << Mangled.substr(0, Len);
  Mangled.remove_prefix(Len);
}

Demangler::Demangler(std::string_view Mangled)
    : Str(Mangled), LastBackref(Mangled.length()) {}

const char *Demangler::parseMangle(OutputBuffer *Demangled) {
  std::string_view M(this->Str);
  parseMangle(Demangled, M, true);
  return M.data();
}

char *llvm::dlangDemangle(std::string_view MangledName) {
  if (MangledName.empty() || !starts_with(MangledName, "_D"))
    return nullptr;

  OutputBuffer Demangled;
  if (MangledName == "_Dmain") {
    Demangled << "D main";
  } else {

    Demangler D(MangledName);
    const char *M = D.parseMangle(&Demangled);

    // Check that the entire symbol was successfully demangled.
    if (M == nullptr || *M != '\0') {
      std::free(Demangled.getBuffer());
      return nullptr;
    }
  }

  // OutputBuffer's internal buffer is not null terminated and therefore we need
  // to add it to comply with C null terminated strings.
  if (Demangled.getCurrentPosition() > 0) {
    Demangled << '\0';
    Demangled.setCurrentPosition(Demangled.getCurrentPosition() - 1);
    return Demangled.getBuffer();
  }

  std::free(Demangled.getBuffer());
  return nullptr;
}
