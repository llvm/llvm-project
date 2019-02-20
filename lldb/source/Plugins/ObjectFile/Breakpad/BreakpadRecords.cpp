//===-- BreakpadRecords.cpp ----------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectFile/Breakpad/BreakpadRecords.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FormatVariadic.h"

using namespace lldb_private;
using namespace lldb_private::breakpad;

namespace {
enum class Token { Unknown, Module, Info, CodeID, File, Func, Public, Stack };
}

static Token toToken(llvm::StringRef str) {
  return llvm::StringSwitch<Token>(str)
      .Case("MODULE", Token::Module)
      .Case("INFO", Token::Info)
      .Case("CODE_ID", Token::CodeID)
      .Case("FILE", Token::File)
      .Case("FUNC", Token::Func)
      .Case("PUBLIC", Token::Public)
      .Case("STACK", Token::Stack)
      .Default(Token::Unknown);
}

static llvm::Triple::OSType toOS(llvm::StringRef str) {
  using llvm::Triple;
  return llvm::StringSwitch<Triple::OSType>(str)
      .Case("Linux", Triple::Linux)
      .Case("mac", Triple::MacOSX)
      .Case("windows", Triple::Win32)
      .Default(Triple::UnknownOS);
}

static llvm::Triple::ArchType toArch(llvm::StringRef str) {
  using llvm::Triple;
  return llvm::StringSwitch<Triple::ArchType>(str)
      .Case("arm", Triple::arm)
      .Case("arm64", Triple::aarch64)
      .Case("mips", Triple::mips)
      .Case("ppc", Triple::ppc)
      .Case("ppc64", Triple::ppc64)
      .Case("s390", Triple::systemz)
      .Case("sparc", Triple::sparc)
      .Case("sparcv9", Triple::sparcv9)
      .Case("x86", Triple::x86)
      .Case("x86_64", Triple::x86_64)
      .Default(Triple::UnknownArch);
}

/// Return the number of hex digits needed to encode an (POD) object of a given
/// type.
template <typename T> static constexpr size_t hex_digits() {
  return 2 * sizeof(T);
}

/// Consume the right number of digits from the input StringRef and convert it
/// to the endian-specific integer N. Return true on success.
template <typename T> static bool consume_hex_integer(llvm::StringRef &str, T &N) {
  llvm::StringRef chunk = str.take_front(hex_digits<T>());
  uintmax_t t;
  if (!to_integer(chunk, t, 16))
    return false;
  N = t;
  str = str.drop_front(hex_digits<T>());
  return true;
}

static UUID parseModuleId(llvm::Triple::OSType os, llvm::StringRef str) {
  struct data_t {
    struct uuid_t {
      llvm::support::ulittle32_t part1;
      llvm::support::ulittle16_t part2[2];
      uint8_t part3[8];
    } uuid;
    llvm::support::ulittle32_t age;
  } data;
  static_assert(sizeof(data) == 20, "");
  // The textual module id encoding should be between 33 and 40 bytes long,
  // depending on the size of the age field, which is of variable length.
  // The first three chunks of the id are encoded in big endian, so we need to
  // byte-swap those.
  if (str.size() <= hex_digits<data_t::uuid_t>() ||
      str.size() > hex_digits<data_t>())
    return UUID();
  if (!consume_hex_integer(str, data.uuid.part1))
    return UUID();
  for (auto &t : data.uuid.part2) {
    if (!consume_hex_integer(str, t))
      return UUID();
  }
  for (auto &t : data.uuid.part3) {
    if (!consume_hex_integer(str, t))
      return UUID();
  }
  uint32_t age;
  if (!to_integer(str, age, 16))
    return UUID();
  data.age = age;

  // On non-windows, the age field should always be zero, so we don't include to
  // match the native uuid format of these platforms.
  return UUID::fromData(&data, os == llvm::Triple::Win32 ? sizeof(data)
                                                         : sizeof(data.uuid));
}

Record::Kind Record::classify(llvm::StringRef Line) {
  Token Tok = toToken(getToken(Line).first);
  switch (Tok) {
  case Token::Module:
    return Record::Module;
  case Token::Info:
    return Record::Info;
  case Token::File:
    return Record::File;
  case Token::Func:
    return Record::Func;
  case Token::Public:
    return Record::Public;
  case Token::Stack:
    return Record::Stack;

  case Token::CodeID:
  case Token::Unknown:
    // Optimistically assume that any unrecognised token means this is a line
    // record, those don't have a special keyword and start directly with a
    // hex number. CODE_ID should never be at the start of a line, but if it
    // is, it can be treated the same way as a garbled line record.
    return Record::Line;
  }
  llvm_unreachable("Fully covered switch above!");
}

llvm::Optional<ModuleRecord> ModuleRecord::parse(llvm::StringRef Line) {
  // MODULE Linux x86_64 E5894855C35DCCCCCCCCCCCCCCCCCCCC0 a.out
  llvm::StringRef Str;
  std::tie(Str, Line) = getToken(Line);
  if (toToken(Str) != Token::Module)
    return llvm::None;

  std::tie(Str, Line) = getToken(Line);
  llvm::Triple::OSType OS = toOS(Str);
  if (OS == llvm::Triple::UnknownOS)
    return llvm::None;

  std::tie(Str, Line) = getToken(Line);
  llvm::Triple::ArchType Arch = toArch(Str);
  if (Arch == llvm::Triple::UnknownArch)
    return llvm::None;

  std::tie(Str, Line) = getToken(Line);
  UUID ID = parseModuleId(OS, Str);
  if (!ID)
    return llvm::None;

  return ModuleRecord(OS, Arch, std::move(ID));
}

llvm::raw_ostream &breakpad::operator<<(llvm::raw_ostream &OS,
                                        const ModuleRecord &R) {
  return OS << "MODULE " << llvm::Triple::getOSTypeName(R.OS) << " "
            << llvm::Triple::getArchTypeName(R.Arch) << " "
            << R.ID.GetAsString();
}

llvm::Optional<InfoRecord> InfoRecord::parse(llvm::StringRef Line) {
  // INFO CODE_ID 554889E55DC3CCCCCCCCCCCCCCCCCCCC [a.exe]
  llvm::StringRef Str;
  std::tie(Str, Line) = getToken(Line);
  if (toToken(Str) != Token::Info)
    return llvm::None;

  std::tie(Str, Line) = getToken(Line);
  if (toToken(Str) != Token::CodeID)
    return llvm::None;

  std::tie(Str, Line) = getToken(Line);
  // If we don't have any text following the code ID (e.g. on linux), we should
  // use this as the UUID. Otherwise, we should revert back to the module ID.
  UUID ID;
  if (Line.trim().empty()) {
    if (Str.empty() || ID.SetFromStringRef(Str, Str.size() / 2) != Str.size())
      return llvm::None;
  }
  return InfoRecord(std::move(ID));
}

llvm::raw_ostream &breakpad::operator<<(llvm::raw_ostream &OS,
                                        const InfoRecord &R) {
  return OS << "INFO CODE_ID " << R.ID.GetAsString();
}

llvm::Optional<FileRecord> FileRecord::parse(llvm::StringRef Line) {
  // FILE number name
  llvm::StringRef Str;
  std::tie(Str, Line) = getToken(Line);
  if (toToken(Str) != Token::File)
    return llvm::None;

  size_t Number;
  std::tie(Str, Line) = getToken(Line);
  if (!to_integer(Str, Number))
    return llvm::None;

  llvm::StringRef Name = Line.trim();
  if (Name.empty())
    return llvm::None;

  return FileRecord(Number, Name);
}

llvm::raw_ostream &breakpad::operator<<(llvm::raw_ostream &OS,
                                        const FileRecord &R) {
  return OS << "FILE " << R.Number << " " << R.Name;
}

static bool parsePublicOrFunc(llvm::StringRef Line, bool &Multiple,
                              lldb::addr_t &Address, lldb::addr_t *Size,
                              lldb::addr_t &ParamSize, llvm::StringRef &Name) {
  // PUBLIC [m] address param_size name
  // or
  // FUNC [m] address size param_size name

  Token Tok = Size ? Token::Func : Token::Public;

  llvm::StringRef Str;
  std::tie(Str, Line) = getToken(Line);
  if (toToken(Str) != Tok)
    return false;

  std::tie(Str, Line) = getToken(Line);
  Multiple = Str == "m";

  if (Multiple)
    std::tie(Str, Line) = getToken(Line);
  if (!to_integer(Str, Address, 16))
    return false;

  if (Tok == Token::Func) {
    std::tie(Str, Line) = getToken(Line);
    if (!to_integer(Str, *Size, 16))
      return false;
  }

  std::tie(Str, Line) = getToken(Line);
  if (!to_integer(Str, ParamSize, 16))
    return false;

  Name = Line.trim();
  if (Name.empty())
    return false;

  return true;
}

llvm::Optional<FuncRecord> FuncRecord::parse(llvm::StringRef Line) {
  bool Multiple;
  lldb::addr_t Address, Size, ParamSize;
  llvm::StringRef Name;

  if (parsePublicOrFunc(Line, Multiple, Address, &Size, ParamSize, Name))
    return FuncRecord(Multiple, Address, Size, ParamSize, Name);

  return llvm::None;
}

bool breakpad::operator==(const FuncRecord &L, const FuncRecord &R) {
  return L.Multiple == R.Multiple && L.Address == R.Address &&
         L.Size == R.Size && L.ParamSize == R.ParamSize && L.Name == R.Name;
}
llvm::raw_ostream &breakpad::operator<<(llvm::raw_ostream &OS,
                                        const FuncRecord &R) {
  return OS << llvm::formatv("FUNC {0}{1:x-} {2:x-} {3:x-} {4}",
                             R.Multiple ? "m " : "", R.Address, R.Size,
                             R.ParamSize, R.Name);
}

llvm::Optional<LineRecord> LineRecord::parse(llvm::StringRef Line) {
  lldb::addr_t Address;
  llvm::StringRef Str;
  std::tie(Str, Line) = getToken(Line);
  if (!to_integer(Str, Address, 16))
    return llvm::None;

  lldb::addr_t Size;
  std::tie(Str, Line) = getToken(Line);
  if (!to_integer(Str, Size, 16))
    return llvm::None;

  uint32_t LineNum;
  std::tie(Str, Line) = getToken(Line);
  if (!to_integer(Str, LineNum))
    return llvm::None;

  size_t FileNum;
  std::tie(Str, Line) = getToken(Line);
  if (!to_integer(Str, FileNum))
    return llvm::None;

  return LineRecord(Address, Size, LineNum, FileNum);
}

bool breakpad::operator==(const LineRecord &L, const LineRecord &R) {
  return L.Address == R.Address && L.Size == R.Size && L.LineNum == R.LineNum &&
         L.FileNum == R.FileNum;
}
llvm::raw_ostream &breakpad::operator<<(llvm::raw_ostream &OS,
                                        const LineRecord &R) {
  return OS << llvm::formatv("{0:x-} {1:x-} {2} {3}", R.Address, R.Size,
                             R.LineNum, R.FileNum);
}

llvm::Optional<PublicRecord> PublicRecord::parse(llvm::StringRef Line) {
  bool Multiple;
  lldb::addr_t Address, ParamSize;
  llvm::StringRef Name;

  if (parsePublicOrFunc(Line, Multiple, Address, nullptr, ParamSize, Name))
    return PublicRecord(Multiple, Address, ParamSize, Name);

  return llvm::None;
}

bool breakpad::operator==(const PublicRecord &L, const PublicRecord &R) {
  return L.Multiple == R.Multiple && L.Address == R.Address &&
         L.ParamSize == R.ParamSize && L.Name == R.Name;
}
llvm::raw_ostream &breakpad::operator<<(llvm::raw_ostream &OS,
                                        const PublicRecord &R) {
  return OS << llvm::formatv("PUBLIC {0}{1:x-} {2:x-} {3}",
                             R.Multiple ? "m " : "", R.Address, R.ParamSize,
                             R.Name);
}

llvm::StringRef breakpad::toString(Record::Kind K) {
  switch (K) {
  case Record::Module:
    return "MODULE";
  case Record::Info:
    return "INFO";
  case Record::File:
    return "FILE";
  case Record::Func:
    return "FUNC";
  case Record::Line:
    return "LINE";
  case Record::Public:
    return "PUBLIC";
  case Record::Stack:
    return "STACK";
  }
  llvm_unreachable("Unknown record kind!");
}
