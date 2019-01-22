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

static llvm::StringRef consume_front(llvm::StringRef &str, size_t n) {
  llvm::StringRef result = str.take_front(n);
  str = str.drop_front(n);
  return result;
}

static UUID parseModuleId(llvm::Triple::OSType os, llvm::StringRef str) {
  struct uuid_data {
    llvm::support::ulittle32_t uuid1;
    llvm::support::ulittle16_t uuid2[2];
    uint8_t uuid3[8];
    llvm::support::ulittle32_t age;
  } data;
  static_assert(sizeof(data) == 20, "");
  // The textual module id encoding should be between 33 and 40 bytes long,
  // depending on the size of the age field, which is of variable length.
  // The first three chunks of the id are encoded in big endian, so we need to
  // byte-swap those.
  if (str.size() < 33 || str.size() > 40)
    return UUID();
  uint32_t t;
  if (to_integer(consume_front(str, 8), t, 16))
    data.uuid1 = t;
  else
    return UUID();
  for (int i = 0; i < 2; ++i) {
    if (to_integer(consume_front(str, 4), t, 16))
      data.uuid2[i] = t;
    else
      return UUID();
  }
  for (int i = 0; i < 8; ++i) {
    if (!to_integer(consume_front(str, 2), data.uuid3[i], 16))
      return UUID();
  }
  if (to_integer(str, t, 16))
    data.age = t;
  else
    return UUID();

  // On non-windows, the age field should always be zero, so we don't include to
  // match the native uuid format of these platforms.
  return UUID::fromData(&data, os == llvm::Triple::Win32 ? 20 : 16);
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

bool breakpad::operator==(const ModuleRecord &L, const ModuleRecord &R) {
  return L.getOS() == R.getOS() && L.getArch() == R.getArch() &&
         L.getID() == R.getID();
}
llvm::raw_ostream &breakpad::operator<<(llvm::raw_ostream &OS,
                                        const ModuleRecord &R) {
  return OS << "MODULE " << llvm::Triple::getOSTypeName(R.getOS()) << " "
            << llvm::Triple::getArchTypeName(R.getArch()) << " "
            << R.getID().GetAsString();
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
  return OS << "INFO CODE_ID " << R.getID().GetAsString();
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
  return L.getMultiple() == R.getMultiple() &&
         L.getAddress() == R.getAddress() && L.getSize() == R.getSize() &&
         L.getParamSize() == R.getParamSize() && L.getName() == R.getName();
}
llvm::raw_ostream &breakpad::operator<<(llvm::raw_ostream &OS,
                                        const FuncRecord &R) {
  return OS << llvm::formatv("FUNC {0}{1:x-} {2:x-} {3:x-} {4}",
                             R.getMultiple() ? "m " : "", R.getAddress(),
                             R.getSize(), R.getParamSize(), R.getName());
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
  return L.getMultiple() == R.getMultiple() &&
         L.getAddress() == R.getAddress() &&
         L.getParamSize() == R.getParamSize() && L.getName() == R.getName();
}
llvm::raw_ostream &breakpad::operator<<(llvm::raw_ostream &OS,
                                        const PublicRecord &R) {
  return OS << llvm::formatv("PUBLIC {0}{1:x-} {2:x-} {3}",
                             R.getMultiple() ? "m " : "", R.getAddress(),
                             R.getParamSize(), R.getName());
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
