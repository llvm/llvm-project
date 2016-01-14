//===-- StreamWriter.h ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_READOBJ_STREAMWRITER_H
#define LLVM_TOOLS_LLVM_READOBJ_STREAMWRITER_H

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace llvm;

namespace llvm {

template<typename T>
struct EnumEntry {
  StringRef Name;
  T Value;
};

struct HexNumber {
  // To avoid sign-extension we have to explicitly cast to the appropriate
  // unsigned type. The overloads are here so that every type that is implicitly
  // convertible to an integer (including enums and endian helpers) can be used
  // without requiring type traits or call-site changes.
  HexNumber(char             Value) : Value(static_cast<unsigned char>(Value)) { }
  HexNumber(signed char      Value) : Value(static_cast<unsigned char>(Value)) { }
  HexNumber(signed short     Value) : Value(static_cast<unsigned short>(Value)) { }
  HexNumber(signed int       Value) : Value(static_cast<unsigned int>(Value)) { }
  HexNumber(signed long      Value) : Value(static_cast<unsigned long>(Value)) { }
  HexNumber(signed long long Value) : Value(static_cast<unsigned long long>(Value)) { }
  HexNumber(unsigned char      Value) : Value(Value) { }
  HexNumber(unsigned short     Value) : Value(Value) { }
  HexNumber(unsigned int       Value) : Value(Value) { }
  HexNumber(unsigned long      Value) : Value(Value) { }
  HexNumber(unsigned long long Value) : Value(Value) { }
  uint64_t Value;
};

raw_ostream &operator<<(raw_ostream &OS, const HexNumber& Value);

class StreamWriter {
public:
  StreamWriter(raw_ostream &OS)
    : OS(OS)
    , IndentLevel(0) {
  }

  void flush() {
    OS.flush();
  }

  void indent(int Levels = 1) {
    IndentLevel += Levels;
  }

  void unindent(int Levels = 1) {
    IndentLevel = std::max(0, IndentLevel - Levels);
  }

  void printIndent() {
    for (int i = 0; i < IndentLevel; ++i)
      OS << "  ";
  }

  template<typename T>
  HexNumber hex(T Value) {
    return HexNumber(Value);
  }

  template<typename T, typename TEnum>
  void printEnum(StringRef Label, T Value,
                 ArrayRef<EnumEntry<TEnum> > EnumValues) {
    StringRef Name;
    bool Found = false;
    for (const auto &EnumItem : EnumValues) {
      if (EnumItem.Value == Value) {
        Name = EnumItem.Name;
        Found = true;
        break;
      }
    }

    if (Found) {
      startLine() << Label << ": " << Name << " (" << hex(Value) << ")\n";
    } else {
      startLine() << Label << ": " << hex(Value) << "\n";
    }
  }

  template <typename T, typename TFlag>
  void printFlags(StringRef Label, T Value, ArrayRef<EnumEntry<TFlag>> Flags,
                  TFlag EnumMask1 = {}, TFlag EnumMask2 = {},
                  TFlag EnumMask3 = {}) {
    typedef EnumEntry<TFlag> FlagEntry;
    typedef SmallVector<FlagEntry, 10> FlagVector;
    FlagVector SetFlags;

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
          (IsEnum  && (Value & EnumMask) == Flag.Value)) {
        SetFlags.push_back(Flag);
      }
    }

    std::sort(SetFlags.begin(), SetFlags.end(), &flagName<TFlag>);

    startLine() << Label << " [ (" << hex(Value) << ")\n";
    for (const auto &Flag : SetFlags) {
      startLine() << "  " << Flag.Name << " (" << hex(Flag.Value) << ")\n";
    }
    startLine() << "]\n";
  }

  template<typename T>
  void printFlags(StringRef Label, T Value) {
    startLine() << Label << " [ (" << hex(Value) << ")\n";
    uint64_t Flag = 1;
    uint64_t Curr = Value;
    while (Curr > 0) {
      if (Curr & 1)
        startLine() << "  " << hex(Flag) << "\n";
      Curr >>= 1;
      Flag <<= 1;
    }
    startLine() << "]\n";
  }

  void printNumber(StringRef Label, uint64_t Value) {
    startLine() << Label << ": " << Value << "\n";
  }

  void printNumber(StringRef Label, uint32_t Value) {
    startLine() << Label << ": " << Value << "\n";
  }

  void printNumber(StringRef Label, uint16_t Value) {
    startLine() << Label << ": " << Value << "\n";
  }

  void printNumber(StringRef Label, uint8_t Value) {
    startLine() << Label << ": " << unsigned(Value) << "\n";
  }

  void printNumber(StringRef Label, int64_t Value) {
    startLine() << Label << ": " << Value << "\n";
  }

  void printNumber(StringRef Label, int32_t Value) {
    startLine() << Label << ": " << Value << "\n";
  }

  void printNumber(StringRef Label, int16_t Value) {
    startLine() << Label << ": " << Value << "\n";
  }

  void printNumber(StringRef Label, int8_t Value) {
    startLine() << Label << ": " << int(Value) << "\n";
  }

  void printNumber(StringRef Label, APSInt Value) {
    startLine() << Label << ": " << Value << "\n";
  }

  void printBoolean(StringRef Label, bool Value) {
    startLine() << Label << ": " << (Value ? "Yes" : "No") << '\n';
  }

  template <typename T>
  void printList(StringRef Label, const T &List) {
    startLine() << Label << ": [";
    bool Comma = false;
    for (const auto &Item : List) {
      if (Comma)
        OS << ", ";
      OS << Item;
      Comma = true;
    }
    OS << "]\n";
  }

  template <typename T>
  void printHexList(StringRef Label, const T &List) {
    startLine() << Label << ": [";
    bool Comma = false;
    for (const auto &Item : List) {
      if (Comma)
        OS << ", ";
      OS << hex(Item);
      Comma = true;
    }
    OS << "]\n";
  }

  template<typename T>
  void printHex(StringRef Label, T Value) {
    startLine() << Label << ": " << hex(Value) << "\n";
  }

  template<typename T>
  void printHex(StringRef Label, StringRef Str, T Value) {
    startLine() << Label << ": " << Str << " (" << hex(Value) << ")\n";
  }

  void printString(StringRef Label, StringRef Value) {
    startLine() << Label << ": " << Value << "\n";
  }

  void printString(StringRef Label, const std::string &Value) {
    startLine() << Label << ": " << Value << "\n";
  }

  template<typename T>
  void printNumber(StringRef Label, StringRef Str, T Value) {
    startLine() << Label << ": " << Str << " (" << Value << ")\n";
  }

  void printBinary(StringRef Label, StringRef Str, ArrayRef<uint8_t> Value) {
    printBinaryImpl(Label, Str, Value, false);
  }

  void printBinary(StringRef Label, StringRef Str, ArrayRef<char> Value) {
    auto V = makeArrayRef(reinterpret_cast<const uint8_t*>(Value.data()),
                          Value.size());
    printBinaryImpl(Label, Str, V, false);
  }

  void printBinary(StringRef Label, ArrayRef<uint8_t> Value) {
    printBinaryImpl(Label, StringRef(), Value, false);
  }

  void printBinary(StringRef Label, ArrayRef<char> Value) {
    auto V = makeArrayRef(reinterpret_cast<const uint8_t*>(Value.data()),
                          Value.size());
    printBinaryImpl(Label, StringRef(), V, false);
  }

  void printBinary(StringRef Label, StringRef Value) {
    auto V = makeArrayRef(reinterpret_cast<const uint8_t*>(Value.data()),
                          Value.size());
    printBinaryImpl(Label, StringRef(), V, false);
  }

  void printBinaryBlock(StringRef Label, StringRef Value) {
    auto V = makeArrayRef(reinterpret_cast<const uint8_t*>(Value.data()),
                          Value.size());
    printBinaryImpl(Label, StringRef(), V, true);
  }

  raw_ostream& startLine() {
    printIndent();
    return OS;
  }

  raw_ostream& getOStream() {
    return OS;
  }

private:
  template<typename T>
  static bool flagName(const EnumEntry<T>& lhs, const EnumEntry<T>& rhs) {
    return lhs.Name < rhs.Name;
  }

  void printBinaryImpl(StringRef Label, StringRef Str, ArrayRef<uint8_t> Value,
                       bool Block);

  raw_ostream &OS;
  int IndentLevel;
};

template <>
inline void
StreamWriter::printHex<support::ulittle16_t>(StringRef Label,
                                             support::ulittle16_t Value) {
  startLine() << Label << ": " << hex(Value) << "\n";
}

struct DictScope {
  DictScope(StreamWriter& W, StringRef N) : W(W) {
    W.startLine() << N << " {\n";
    W.indent();
  }

  ~DictScope() {
    W.unindent();
    W.startLine() << "}\n";
  }

  StreamWriter& W;
};

struct ListScope {
  ListScope(StreamWriter& W, StringRef N) : W(W) {
    W.startLine() << N << " [\n";
    W.indent();
  }

  ~ListScope() {
    W.unindent();
    W.startLine() << "]\n";
  }

  StreamWriter& W;
};

} // namespace llvm

#endif
