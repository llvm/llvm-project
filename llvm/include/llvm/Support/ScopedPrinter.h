//===-- ScopedPrinter.h ----------------------------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_SCOPEDPRINTER_H
#define LLVM_SUPPORT_SCOPEDPRINTER_H

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace llvm {

template <typename T> struct EnumEntry {
  StringRef Name;
  // While Name suffices in most of the cases, in certain cases
  // GNU style and LLVM style of ELFDumper do not
  // display same string for same enum. The AltName if initialized appropriately
  // will hold the string that GNU style emits.
  // Example:
  // "EM_X86_64" string on LLVM style for Elf_Ehdr->e_machine corresponds to
  // "Advanced Micro Devices X86-64" on GNU style
  StringRef AltName;
  T Value;
  constexpr EnumEntry(StringRef N, StringRef A, T V)
      : Name(N), AltName(A), Value(V) {}
  constexpr EnumEntry(StringRef N, T V) : Name(N), AltName(N), Value(V) {}
};

struct HexNumber {
  // To avoid sign-extension we have to explicitly cast to the appropriate
  // unsigned type. The overloads are here so that every type that is implicitly
  // convertible to an integer (including enums and endian helpers) can be used
  // without requiring type traits or call-site changes.
  HexNumber(char Value) : Value(static_cast<unsigned char>(Value)) {}
  HexNumber(signed char Value) : Value(static_cast<unsigned char>(Value)) {}
  HexNumber(signed short Value) : Value(static_cast<unsigned short>(Value)) {}
  HexNumber(signed int Value) : Value(static_cast<unsigned int>(Value)) {}
  HexNumber(signed long Value) : Value(static_cast<unsigned long>(Value)) {}
  HexNumber(signed long long Value)
      : Value(static_cast<unsigned long long>(Value)) {}
  HexNumber(unsigned char Value) : Value(Value) {}
  HexNumber(unsigned short Value) : Value(Value) {}
  HexNumber(unsigned int Value) : Value(Value) {}
  HexNumber(unsigned long Value) : Value(Value) {}
  HexNumber(unsigned long long Value) : Value(Value) {}
  uint64_t Value;
};

struct FlagEntry {
  FlagEntry(StringRef Name, char Value)
      : Name(Name), Value(static_cast<unsigned char>(Value)) {}
  FlagEntry(StringRef Name, signed char Value)
      : Name(Name), Value(static_cast<unsigned char>(Value)) {}
  FlagEntry(StringRef Name, signed short Value)
      : Name(Name), Value(static_cast<unsigned short>(Value)) {}
  FlagEntry(StringRef Name, signed int Value)
      : Name(Name), Value(static_cast<unsigned int>(Value)) {}
  FlagEntry(StringRef Name, signed long Value)
      : Name(Name), Value(static_cast<unsigned long>(Value)) {}
  FlagEntry(StringRef Name, signed long long Value)
      : Name(Name), Value(static_cast<unsigned long long>(Value)) {}
  FlagEntry(StringRef Name, unsigned char Value) : Name(Name), Value(Value) {}
  FlagEntry(StringRef Name, unsigned short Value) : Name(Name), Value(Value) {}
  FlagEntry(StringRef Name, unsigned int Value) : Name(Name), Value(Value) {}
  FlagEntry(StringRef Name, unsigned long Value) : Name(Name), Value(Value) {}
  FlagEntry(StringRef Name, unsigned long long Value)
      : Name(Name), Value(Value) {}
  StringRef Name;
  uint64_t Value;
};

raw_ostream &operator<<(raw_ostream &OS, const HexNumber &Value);
std::string to_hexString(uint64_t Value, bool UpperCase = true);

template <class T> std::string to_string(const T &Value) {
  std::string number;
  raw_string_ostream stream(number);
  stream << Value;
  return stream.str();
}

template <typename T, typename TEnum>
std::string enumToString(T Value, ArrayRef<EnumEntry<TEnum>> EnumValues) {
  for (const EnumEntry<TEnum> &EnumItem : EnumValues)
    if (EnumItem.Value == Value)
      return std::string(EnumItem.AltName);
  return to_hexString(Value, false);
}

class ScopedPrinter {
public:
  ScopedPrinter(raw_ostream &OS) : OS(OS), IndentLevel(0) {}

  virtual ~ScopedPrinter() {}

  void flush() { OS.flush(); }

  void indent(int Levels = 1) { IndentLevel += Levels; }

  void unindent(int Levels = 1) {
    IndentLevel = std::max(0, IndentLevel - Levels);
  }

  void resetIndent() { IndentLevel = 0; }

  int getIndentLevel() { return IndentLevel; }

  void setPrefix(StringRef P) { Prefix = P; }

  void printIndent() {
    OS << Prefix;
    for (int i = 0; i < IndentLevel; ++i)
      OS << "  ";
  }

  template <typename T> HexNumber hex(T Value) { return HexNumber(Value); }

  template <typename T, typename TEnum>
  void printEnum(StringRef Label, T Value,
                 ArrayRef<EnumEntry<TEnum>> EnumValues) {
    StringRef Name;
    bool Found = false;
    for (const auto &EnumItem : EnumValues) {
      if (EnumItem.Value == Value) {
        Name = EnumItem.Name;
        Found = true;
        break;
      }
    }

    if (Found)
      printHex(Label, Name, Value);
    else
      printHex(Label, Value);
  }

  template <typename T, typename TFlag>
  void printFlags(StringRef Label, T Value, ArrayRef<EnumEntry<TFlag>> Flags,
                  TFlag EnumMask1 = {}, TFlag EnumMask2 = {},
                  TFlag EnumMask3 = {}) {
    SmallVector<FlagEntry, 10> SetFlags;

    for (const auto &Flag : Flags) {
      if (Flag.Value == 0)
        continue;

      TFlag EnumMask{};
      if (Flag.Value & EnumMask1)
        EnumMask = EnumMask1;
      else if (Flag.Value & EnumMask2)
        EnumMask = EnumMask2;
      else if (Flag.Value & EnumMask3)
        EnumMask = EnumMask3;
      bool IsEnum = (Flag.Value & EnumMask) != 0;
      if ((!IsEnum && (Value & Flag.Value) == Flag.Value) ||
          (IsEnum && (Value & EnumMask) == Flag.Value)) {
        SetFlags.emplace_back(Flag.Name, Flag.Value);
      }
    }

    llvm::sort(SetFlags, &flagName);
    printFlagsImpl(Label, hex(Value), SetFlags);
  }

  template <typename T> void printFlags(StringRef Label, T Value) {
    SmallVector<HexNumber, 10> SetFlags;
    uint64_t Flag = 1;
    uint64_t Curr = Value;
    while (Curr > 0) {
      if (Curr & 1)
        SetFlags.emplace_back(Flag);
      Curr >>= 1;
      Flag <<= 1;
    }
    printFlagsImpl(Label, hex(Value), SetFlags);
  }

  virtual void printNumber(StringRef Label, uint64_t Value) {
    startLine() << Label << ": " << Value << "\n";
  }

  virtual void printNumber(StringRef Label, uint32_t Value) {
    startLine() << Label << ": " << Value << "\n";
  }

  virtual void printNumber(StringRef Label, uint16_t Value) {
    startLine() << Label << ": " << Value << "\n";
  }

  virtual void printNumber(StringRef Label, uint8_t Value) {
    startLine() << Label << ": " << unsigned(Value) << "\n";
  }

  virtual void printNumber(StringRef Label, int64_t Value) {
    startLine() << Label << ": " << Value << "\n";
  }

  virtual void printNumber(StringRef Label, int32_t Value) {
    startLine() << Label << ": " << Value << "\n";
  }

  virtual void printNumber(StringRef Label, int16_t Value) {
    startLine() << Label << ": " << Value << "\n";
  }

  virtual void printNumber(StringRef Label, int8_t Value) {
    startLine() << Label << ": " << int(Value) << "\n";
  }

  virtual void printNumber(StringRef Label, const APSInt &Value) {
    startLine() << Label << ": " << Value << "\n";
  }

  template <typename T>
  void printNumber(StringRef Label, StringRef Str, T Value) {
    printNumberImpl(Label, Str, to_string(Value));
  }

  virtual void printBoolean(StringRef Label, bool Value) {
    startLine() << Label << ": " << (Value ? "Yes" : "No") << '\n';
  }

  template <typename... T> void printVersion(StringRef Label, T... Version) {
    startLine() << Label << ": ";
    printVersionInternal(Version...);
    getOStream() << "\n";
  }

  template <typename T>
  void printList(StringRef Label, const ArrayRef<T> List) {
    SmallVector<std::string, 10> StringList;
    for (const auto &Item : List)
      StringList.emplace_back(to_string(Item));
    printList(Label, StringList);
  }

  virtual void printList(StringRef Label, const ArrayRef<bool> List) {
    printListImpl(Label, List);
  }

  virtual void printList(StringRef Label, const ArrayRef<std::string> List) {
    printListImpl(Label, List);
  }

  virtual void printList(StringRef Label, const ArrayRef<uint64_t> List) {
    printListImpl(Label, List);
  }

  virtual void printList(StringRef Label, const ArrayRef<uint32_t> List) {
    printListImpl(Label, List);
  }

  virtual void printList(StringRef Label, const ArrayRef<uint16_t> List) {
    printListImpl(Label, List);
  }

  virtual void printList(StringRef Label, const ArrayRef<uint8_t> List) {
    SmallVector<unsigned> NumberList;
    for (const uint8_t &Item : List)
      NumberList.emplace_back(Item);
    printListImpl(Label, NumberList);
  }

  virtual void printList(StringRef Label, const ArrayRef<int64_t> List) {
    printListImpl(Label, List);
  }

  virtual void printList(StringRef Label, const ArrayRef<int32_t> List) {
    printListImpl(Label, List);
  }

  virtual void printList(StringRef Label, const ArrayRef<int16_t> List) {
    printListImpl(Label, List);
  }

  virtual void printList(StringRef Label, const ArrayRef<int8_t> List) {
    SmallVector<int> NumberList;
    for (const int8_t &Item : List)
      NumberList.emplace_back(Item);
    printListImpl(Label, NumberList);
  }

  virtual void printList(StringRef Label, const ArrayRef<APSInt> List) {
    printListImpl(Label, List);
  }

  template <typename T, typename U>
  void printList(StringRef Label, const T &List, const U &Printer) {
    startLine() << Label << ": [";
    ListSeparator LS;
    for (const auto &Item : List) {
      OS << LS;
      Printer(OS, Item);
    }
    OS << "]\n";
  }

  template <typename T> void printHexList(StringRef Label, const T &List) {
    SmallVector<HexNumber> HexList;
    for (const auto &Item : List)
      HexList.emplace_back(Item);
    printHexListImpl(Label, HexList);
  }

  template <typename T> void printHex(StringRef Label, T Value) {
    printHexImpl(Label, hex(Value));
  }

  template <typename T> void printHex(StringRef Label, StringRef Str, T Value) {
    printHexImpl(Label, Str, hex(Value));
  }

  template <typename T>
  void printSymbolOffset(StringRef Label, StringRef Symbol, T Value) {
    printSymbolOffsetImpl(Label, Symbol, hex(Value));
  }

  virtual void printString(StringRef Value) { startLine() << Value << "\n"; }

  virtual void printString(StringRef Label, StringRef Value) {
    startLine() << Label << ": " << Value << "\n";
  }

  void printBinary(StringRef Label, StringRef Str, ArrayRef<uint8_t> Value) {
    printBinaryImpl(Label, Str, Value, false);
  }

  void printBinary(StringRef Label, StringRef Str, ArrayRef<char> Value) {
    auto V = makeArrayRef(reinterpret_cast<const uint8_t *>(Value.data()),
                          Value.size());
    printBinaryImpl(Label, Str, V, false);
  }

  void printBinary(StringRef Label, ArrayRef<uint8_t> Value) {
    printBinaryImpl(Label, StringRef(), Value, false);
  }

  void printBinary(StringRef Label, ArrayRef<char> Value) {
    auto V = makeArrayRef(reinterpret_cast<const uint8_t *>(Value.data()),
                          Value.size());
    printBinaryImpl(Label, StringRef(), V, false);
  }

  void printBinary(StringRef Label, StringRef Value) {
    auto V = makeArrayRef(reinterpret_cast<const uint8_t *>(Value.data()),
                          Value.size());
    printBinaryImpl(Label, StringRef(), V, false);
  }

  void printBinaryBlock(StringRef Label, ArrayRef<uint8_t> Value,
                        uint32_t StartOffset) {
    printBinaryImpl(Label, StringRef(), Value, true, StartOffset);
  }

  void printBinaryBlock(StringRef Label, ArrayRef<uint8_t> Value) {
    printBinaryImpl(Label, StringRef(), Value, true);
  }

  void printBinaryBlock(StringRef Label, StringRef Value) {
    auto V = makeArrayRef(reinterpret_cast<const uint8_t *>(Value.data()),
                          Value.size());
    printBinaryImpl(Label, StringRef(), V, true);
  }

  template <typename T> void printObject(StringRef Label, const T &Value) {
    printString(Label, to_string(Value));
  }

  virtual void objectBegin() { scopedBegin('{'); }

  virtual void objectBegin(StringRef Label) { scopedBegin(Label, '{'); }

  virtual void objectEnd() { scopedEnd('}'); }

  virtual void arrayBegin() { scopedBegin('['); }

  virtual void arrayBegin(StringRef Label) { scopedBegin(Label, '['); }

  virtual void arrayEnd() { scopedEnd(']'); }

  virtual raw_ostream &startLine() {
    printIndent();
    return OS;
  }

  virtual raw_ostream &getOStream() { return OS; }

private:
  template <typename T> void printVersionInternal(T Value) {
    getOStream() << Value;
  }

  template <typename S, typename T, typename... TArgs>
  void printVersionInternal(S Value, T Value2, TArgs... Args) {
    getOStream() << Value << ".";
    printVersionInternal(Value2, Args...);
  }

  static bool flagName(const FlagEntry &LHS, const FlagEntry &RHS) {
    return LHS.Name < RHS.Name;
  }

  virtual void printBinaryImpl(StringRef Label, StringRef Str,
                               ArrayRef<uint8_t> Value, bool Block,
                               uint32_t StartOffset = 0);

  virtual void printFlagsImpl(StringRef Label, HexNumber Value,
                              ArrayRef<FlagEntry> Flags) {
    startLine() << Label << " [ (" << Value << ")\n";
    for (const auto &Flag : Flags)
      startLine() << "  " << Flag.Name << " (" << hex(Flag.Value) << ")\n";
    startLine() << "]\n";
  }

  virtual void printFlagsImpl(StringRef Label, HexNumber Value,
                              ArrayRef<HexNumber> Flags) {
    startLine() << Label << " [ (" << Value << ")\n";
    for (const auto &Flag : Flags)
      startLine() << "  " << Flag << '\n';
    startLine() << "]\n";
  }

  template <typename T> void printListImpl(StringRef Label, const T List) {
    startLine() << Label << ": [";
    ListSeparator LS;
    for (const auto &Item : List)
      OS << LS << Item;
    OS << "]\n";
  }

  virtual void printHexListImpl(StringRef Label,
                                const ArrayRef<HexNumber> List) {
    startLine() << Label << ": [";
    ListSeparator LS;
    for (const auto &Item : List)
      OS << LS << hex(Item);
    OS << "]\n";
  }

  virtual void printHexImpl(StringRef Label, HexNumber Value) {
    startLine() << Label << ": " << Value << "\n";
  }

  virtual void printHexImpl(StringRef Label, StringRef Str, HexNumber Value) {
    startLine() << Label << ": " << Str << " (" << Value << ")\n";
  }

  virtual void printSymbolOffsetImpl(StringRef Label, StringRef Symbol,
                                     HexNumber Value) {
    startLine() << Label << ": " << Symbol << '+' << Value << '\n';
  }

  virtual void printNumberImpl(StringRef Label, StringRef Str,
                               StringRef Value) {
    startLine() << Label << ": " << Str << " (" << Value << ")\n";
  }

  void scopedBegin(char Symbol) {
    startLine() << Symbol << '\n';
    indent();
  }

  void scopedBegin(StringRef Label, char Symbol) {
    startLine() << Label;
    if (!Label.empty())
      OS << ' ';
    OS << Symbol << '\n';
    indent();
  }

  void scopedEnd(char Symbol) {
    unindent();
    startLine() << Symbol << '\n';
  }

  raw_ostream &OS;
  int IndentLevel;
  StringRef Prefix;
};

template <>
inline void
ScopedPrinter::printHex<support::ulittle16_t>(StringRef Label,
                                              support::ulittle16_t Value) {
  startLine() << Label << ": " << hex(Value) << "\n";
}

struct DelimitedScope {
  DelimitedScope(ScopedPrinter &W) : W(W) {}
  virtual ~DelimitedScope(){};
  ScopedPrinter &W;
};

struct DictScope : DelimitedScope {
  explicit DictScope(ScopedPrinter &W) : DelimitedScope(W) { W.objectBegin(); }

  DictScope(ScopedPrinter &W, StringRef N) : DelimitedScope(W) {
    W.objectBegin(N);
  }

  ~DictScope() { W.objectEnd(); }
};

struct ListScope : DelimitedScope {
  explicit ListScope(ScopedPrinter &W) : DelimitedScope(W) { W.arrayBegin(); }

  ListScope(ScopedPrinter &W, StringRef N) : DelimitedScope(W) {
    W.arrayBegin(N);
  }

  ~ListScope() { W.arrayEnd(); }
};

} // namespace llvm

#endif
