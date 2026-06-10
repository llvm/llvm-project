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
  ///
  /// \see https://dlang.org/spec/abi.html#name_mangling .
  /// \see https://dlang.org/spec/abi.html#MangledName .
  void parseMangle(OutputBuffer *Demangled, std::string_view &Mangled);

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
  /// \param Mangled mangled symbol to be demangled.
  ///
  /// \return true on success, false on error.
  ///
  /// \see https://dlang.org/spec/abi.html#Type .
  bool parseType(OutputBuffer *Demangled, std::string_view &Mangled);

  /// An immutable view of the string we are demangling.
  const std::string_view Str;
  /// The index of the last back reference.
  int LastBackref;
};

} // namespace

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

  // TODO: Handle template instances.

  if (Mangled.front() != 'Q')
    return false;

  Mangled.remove_prefix(1);
  bool Valid = decodeBackrefPos(Mangled, Ret);
  if (!Valid || Ret > Qref - Str.data())
    return false;

  return std::isdigit(Qref[-Ret]);
}

void Demangler::parseMangle(OutputBuffer *Demangled,
                            std::string_view &Mangled) {
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

  parseQualified(Demangled, Mangled);

  if (Mangled.empty()) {
    Mangled = {};
    return;
  }

  // Artificial symbols end with 'Z' and have no type.
  if (Mangled.front() == 'Z') {
    Mangled.remove_prefix(1);
  } else {
    OutputBuffer TypeBuf;
    if (!parseType(&TypeBuf, Mangled))
      Mangled = {};
    else {
      // Insert type before the symbol name that was already appended.
      size_t TypeLen = TypeBuf.getCurrentPosition();
      if (TypeLen > 0) {
        Demangled->insert(0, TypeBuf.getBuffer(), TypeLen);
        Demangled->insert(TypeLen, " ", 1);
      }
    }
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

  // TODO: Parse lengthless template instances.

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
  // TODO: Parse type qualifiers.
  // TODO: Parse function types.
  // TODO: Parse compound types.
  // TODO: Parse delegate types.
  // TODO: Parse tuple types.

  // Basic types.
  case 'i':
    Mangled.remove_prefix(1);
    *Demangled << "int";
    return true;

  case 'v':
    Mangled.remove_prefix(1);
    *Demangled << "void";
    return true;

  case 'a':
    Mangled.remove_prefix(1);
    *Demangled << "char";
    return true;
  case 'b':
    Mangled.remove_prefix(1);
    *Demangled << "bool";
    return true;
  case 'c':
    Mangled.remove_prefix(1);
    *Demangled << "creal";
    return true;
  case 'd':
    Mangled.remove_prefix(1);
    *Demangled << "double";
    return true;
  case 'e':
    Mangled.remove_prefix(1);
    *Demangled << "real";
    return true;
  case 'f':
    Mangled.remove_prefix(1);
    *Demangled << "float";
    return true;
  case 'g':
    Mangled.remove_prefix(1);
    *Demangled << "byte";
    return true;
  case 'h':
    Mangled.remove_prefix(1);
    *Demangled << "ubyte";
    return true;
  case 'j':
    Mangled.remove_prefix(1);
    *Demangled << "ireal";
    return true;
  case 'k':
    Mangled.remove_prefix(1);
    *Demangled << "uint";
    return true;
  case 'l':
    Mangled.remove_prefix(1);
    *Demangled << "long";
    return true;
  case 'm':
    Mangled.remove_prefix(1);
    *Demangled << "ulong";
    return true;
  case 'n':
    Mangled.remove_prefix(1);
    return true; // typeof(null), no output
  case 'o':
    Mangled.remove_prefix(1);
    *Demangled << "ifloat";
    return true;
  case 'p':
    Mangled.remove_prefix(1);
    *Demangled << "idouble";
    return true;
  case 'q':
    Mangled.remove_prefix(1);
    *Demangled << "cfloat";
    return true;
  case 'r':
    Mangled.remove_prefix(1);
    *Demangled << "cdouble";
    return true;
  case 's':
    Mangled.remove_prefix(1);
    *Demangled << "short";
    return true;
  case 't':
    Mangled.remove_prefix(1);
    *Demangled << "ushort";
    return true;
  case 'u':
    Mangled.remove_prefix(1);
    *Demangled << "wchar";
    return true;
  case 'w':
    Mangled.remove_prefix(1);
    *Demangled << "dchar";
    return true;
  case 'z': // two-char: zi=cent, zk=ucent
    if (Mangled.size() < 2) {
      Mangled = {};
      return false;
    }
    if (Mangled[1] == 'i') {
      Mangled.remove_prefix(2);
      *Demangled << "cent";
      return true;
    }
    if (Mangled[1] == 'k') {
      Mangled.remove_prefix(2);
      *Demangled << "ucent";
      return true;
    }
    Mangled = {};
    return false;
  case 'N':
    if (Mangled.size() < 2) {
      Mangled = {};
      return false;
    }
    if (Mangled[1] == 'n') {
      Mangled.remove_prefix(2);
      *Demangled << "noreturn";
      return true;
    }
    Mangled = {};
    return false;

  // Back referenced type.
  case 'Q': {
    parseTypeBackref(Demangled, Mangled);
    return true;
  }

  default: // unhandled.
    Mangled = {};
    return false;
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
  parseMangle(Demangled, M);
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
