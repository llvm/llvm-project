//===- llvm/unittests/TextAPI/YAMLTest.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===/

#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/InterfaceStub/IFSHandler.h"
#include "llvm/InterfaceStub/IFSStub.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <string>

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::ifs;

void compareByLine(StringRef LHS, StringRef RHS) {
  StringRef Line1;
  StringRef Line2;
  while (LHS.size() > 0 && RHS.size() > 0) {
    std::tie(Line1, LHS) = LHS.split('\n');
    std::tie(Line2, RHS) = RHS.split('\n');
    // Comparing StringRef objects works, but has messy output when not equal.
    // Using STREQ on StringRef.data() doesn't work since these substrings are
    // not null terminated.
    // This is inefficient, but forces null terminated strings that can be
    // cleanly compared.
    EXPECT_STREQ(Line1.str().data(), Line2.str().data());
  }
}

TEST(ElfYamlTextAPI, YAMLReadableTBE) {
  const char Data[] = "--- !ifs-v1\n"
                      "IfsVersion: 1.0\n"
                      "Target: { ObjectFormat: ELF, Arch: x86_64, Endianness: "
                      "little, BitWidth: 64 }\n"
                      "NeededLibs: [libc.so, libfoo.so, libbar.so]\n"
                      "Symbols:\n"
                      "  - { Name: foo, Type: Func, Undefined: true }\n"
                      "...\n";
  Expected<std::unique_ptr<IFSStub>> StubOrErr = readIFSFromBuffer(Data);
  ASSERT_THAT_ERROR(StubOrErr.takeError(), Succeeded());
  std::unique_ptr<IFSStub> Stub = std::move(StubOrErr.get());
  EXPECT_NE(Stub.get(), nullptr);
  EXPECT_FALSE(Stub->SoName.has_value());
  EXPECT_TRUE(Stub->Target.Arch.has_value());
  EXPECT_EQ(*Stub->Target.Arch, (uint16_t)llvm::ELF::EM_X86_64);
  EXPECT_EQ(Stub->NeededLibs.size(), 3u);
  EXPECT_STREQ(Stub->NeededLibs[0].c_str(), "libc.so");
  EXPECT_STREQ(Stub->NeededLibs[1].c_str(), "libfoo.so");
  EXPECT_STREQ(Stub->NeededLibs[2].c_str(), "libbar.so");
}

TEST(ElfYamlTextAPI, YAMLReadsInvalidSymbols) {
  const char Data[] =
      "--- !ifs-v1\n"
      "IfsVersion: 1.0\n"
      "SoName: test.so\n"
      "Target: { ObjectFormat: ELF, Arch: x86_64, Endianness: little, "
      "BitWidth: 64 }\n"
      "Symbols:\n"
      "  - { Name: not, Type: File, Undefined: true, Size: 111, "
      "Weak: true, Warning: \'All fields populated!\' }\n"
      "...\n";
  Expected<std::unique_ptr<IFSStub>> StubOrErr = readIFSFromBuffer(Data);
  ASSERT_THAT_ERROR(
      StubOrErr.takeError(),
      FailedWithMessage("IFS symbol type for symbol 'not' is unsupported"));
}

TEST(ElfYamlTextAPI, YAMLReadsTBESymbols) {
  const char Data[] =
      "--- !ifs-v1\n"
      "IfsVersion: 1.0\n"
      "SoName: test.so\n"
      "Target: { ObjectFormat: ELF, Arch: x86_64, Endianness: little, "
      "BitWidth: 64 }\n"
      "Symbols:\n"
      "  - { Name: bar, Type: Object, Size: 42 }\n"
      "  - { Name: baz, Type: TLS, Size: 3 }\n"
      "  - { Name: foo, Type: Func, Warning: \"Deprecated!\" }\n"
      "  - { Name: nor, Type: NoType, Undefined: true }\n"
      "  - { Name: not, Type: NoType, Undefined: true, Size: 111, "
      "Weak: true, Warning: \'All fields populated!\' }\n"
      "...\n";
  Expected<std::unique_ptr<IFSStub>> StubOrErr = readIFSFromBuffer(Data);
  ASSERT_THAT_ERROR(StubOrErr.takeError(), Succeeded());
  std::unique_ptr<IFSStub> Stub = std::move(StubOrErr.get());
  EXPECT_NE(Stub.get(), nullptr);
  EXPECT_TRUE(Stub->SoName);
  EXPECT_STREQ(Stub->SoName->c_str(), "test.so");
  EXPECT_EQ(Stub->Symbols.size(), 5u);

  auto Iterator = Stub->Symbols.begin();
  IFSSymbol const &SymBar = *Iterator++;
  EXPECT_STREQ(SymBar.Name.c_str(), "bar");
  EXPECT_EQ(*SymBar.Size, 42u);
  EXPECT_EQ(SymBar.Type, IFSSymbolType::Object);
  EXPECT_FALSE(SymBar.Undefined);
  EXPECT_FALSE(SymBar.Weak);
  EXPECT_FALSE(SymBar.Warning);

  IFSSymbol const &SymBaz = *Iterator++;
  EXPECT_STREQ(SymBaz.Name.c_str(), "baz");
  EXPECT_EQ(*SymBaz.Size, 3u);
  EXPECT_EQ(SymBaz.Type, IFSSymbolType::TLS);
  EXPECT_FALSE(SymBaz.Undefined);
  EXPECT_FALSE(SymBaz.Weak);
  EXPECT_FALSE(SymBaz.Warning.has_value());

  IFSSymbol const &SymFoo = *Iterator++;
  EXPECT_STREQ(SymFoo.Name.c_str(), "foo");
  EXPECT_FALSE(SymFoo.Size.has_value());
  EXPECT_EQ(SymFoo.Type, IFSSymbolType::Func);
  EXPECT_FALSE(SymFoo.Undefined);
  EXPECT_FALSE(SymFoo.Weak);
  EXPECT_TRUE(SymFoo.Warning.has_value());
  EXPECT_STREQ(SymFoo.Warning->c_str(), "Deprecated!");

  IFSSymbol const &SymNor = *Iterator++;
  EXPECT_STREQ(SymNor.Name.c_str(), "nor");
  EXPECT_FALSE(SymNor.Size.has_value());
  EXPECT_EQ(SymNor.Type, IFSSymbolType::NoType);
  EXPECT_TRUE(SymNor.Undefined);
  EXPECT_FALSE(SymNor.Weak);
  EXPECT_FALSE(SymNor.Warning.has_value());

  IFSSymbol const &SymNot = *Iterator++;
  EXPECT_STREQ(SymNot.Name.c_str(), "not");
  EXPECT_EQ(*SymNot.Size, 111u);
  EXPECT_EQ(SymNot.Type, IFSSymbolType::NoType);
  EXPECT_TRUE(SymNot.Undefined);
  EXPECT_TRUE(SymNot.Weak);
  EXPECT_TRUE(SymNot.Warning);
  EXPECT_STREQ(SymNot.Warning->c_str(), "All fields populated!");
}

TEST(ElfYamlTextAPI, YAMLReadsNoTBESyms) {
  const char Data[] = "--- !ifs-v1\n"
                      "IfsVersion: 1.0\n"
                      "SoName: test.so\n"
                      "Target: { ObjectFormat: ELF, Arch: x86_64, Endianness: "
                      "little, BitWidth: 64 }\n"
                      "Symbols: []\n"
                      "...\n";
  Expected<std::unique_ptr<IFSStub>> StubOrErr = readIFSFromBuffer(Data);
  ASSERT_THAT_ERROR(StubOrErr.takeError(), Succeeded());
  std::unique_ptr<IFSStub> Stub = std::move(StubOrErr.get());
  EXPECT_NE(Stub.get(), nullptr);
  EXPECT_EQ(0u, Stub->Symbols.size());
}

TEST(ElfYamlTextAPI, YAMLUnreadableTBE) {
  // Can't read: wrong format/version.
  const char Data[] = "--- !tapi-tbz\n"
                      "IfsVersion: z.3\n"
                      "SoName: test.so\n"
                      "Target: { ObjectFormat: ELF, Arch: x86_64, Endianness: "
                      "little, BitWidth: 64 }\n"
                      "Symbols:\n"
                      "  foo: { Type: Func, Undefined: true }\n";
  Expected<std::unique_ptr<IFSStub>> StubOrErr = readIFSFromBuffer(Data);
  ASSERT_THAT_ERROR(StubOrErr.takeError(), Failed());
}

TEST(ElfYamlTextAPI, YAMLUnsupportedVersion) {
  const char Data[] = "--- !ifs-v1\n"
                      "IfsVersion: 9.9.9\n"
                      "SoName: test.so\n"
                      "Target: { ObjectFormat: ELF, Arch: x86_64, Endianness: "
                      "little, BitWidth: 64 }\n"
                      "Symbols: []\n"
                      "...\n";
  Expected<std::unique_ptr<IFSStub>> StubOrErr = readIFSFromBuffer(Data);
  std::string ErrorMessage = toString(StubOrErr.takeError());
  EXPECT_EQ("IFS version 9.9.9 is unsupported.", ErrorMessage);
}

TEST(ElfYamlTextAPI, YAMLWritesTBESymbols) {
  const char Expected[] =
      "--- !ifs-v1\n"
      "IfsVersion:      1.0\n"
      "Target:          { ObjectFormat: ELF, Arch: AArch64, Endianness: "
      "little, BitWidth: 64 }\n"
      "Symbols:\n"
      "  - { Name: bar, Type: Func, Weak: true }\n"
      "  - { Name: foo, Type: NoType, Size: 99, Warning: Does nothing }\n"
      "  - { Name: nor, Type: Func, Undefined: true }\n"
      "  - { Name: not, Type: Unknown, Size: 12345678901234 }\n"
      "...\n";
  IFSStub Stub;
  Stub.IfsVersion = VersionTuple(1, 0);
  Stub.Target.Arch = ELF::EM_AARCH64;
  Stub.Target.BitWidth = IFSBitWidthType::IFS64;
  Stub.Target.Endianness = IFSEndiannessType::Little;
  Stub.Target.ObjectFormat = "ELF";

  IFSSymbol SymBar("bar");
  SymBar.Size = 128u;
  SymBar.Type = IFSSymbolType::Func;
  SymBar.Undefined = false;
  SymBar.Weak = true;

  IFSSymbol SymFoo("foo");
  SymFoo.Size = 99u;
  SymFoo.Type = IFSSymbolType::NoType;
  SymFoo.Undefined = false;
  SymFoo.Weak = false;
  SymFoo.Warning = "Does nothing";

  IFSSymbol SymNor("nor");
  SymNor.Size = 1234u;
  SymNor.Type = IFSSymbolType::Func;
  SymNor.Undefined = true;
  SymNor.Weak = false;

  IFSSymbol SymNot("not");
  SymNot.Size = 12345678901234u;
  SymNot.Type = IFSSymbolType::Unknown;
  SymNot.Undefined = false;
  SymNot.Weak = false;

  // Symbol order is preserved instead of being sorted.
  Stub.Symbols.push_back(SymBar);
  Stub.Symbols.push_back(SymFoo);
  Stub.Symbols.push_back(SymNor);
  Stub.Symbols.push_back(SymNot);

  // Ensure move constructor works as expected.
  IFSStub Moved = std::move(Stub);

  std::string Result;
  raw_string_ostream OS(Result);
  ASSERT_THAT_ERROR(writeIFSToOutputStream(OS, Moved), Succeeded());
  compareByLine(Result.c_str(), Expected);
}

TEST(ElfYamlTextAPI, YAMLWritesNoTBESyms) {
  const char Expected[] = "--- !ifs-v1\n"
                          "IfsVersion:      1.0\n"
                          "SoName:          nosyms.so\n"
                          "Target:          { ObjectFormat: ELF, Arch: x86_64, "
                          "Endianness: little, BitWidth: 64 }\n"
                          "NeededLibs:\n"
                          "  - libc.so\n"
                          "  - libfoo.so\n"
                          "  - libbar.so\n"
                          "Symbols:         []\n"
                          "...\n";
  IFSStub Stub;
  Stub.IfsVersion = VersionTuple(1, 0);
  Stub.SoName = "nosyms.so";
  Stub.Target.Arch = ELF::EM_X86_64;
  Stub.Target.BitWidth = IFSBitWidthType::IFS64;
  Stub.Target.Endianness = IFSEndiannessType::Little;
  Stub.Target.ObjectFormat = "ELF";
  Stub.NeededLibs.push_back("libc.so");
  Stub.NeededLibs.push_back("libfoo.so");
  Stub.NeededLibs.push_back("libbar.so");

  std::string Result;
  raw_string_ostream OS(Result);
  ASSERT_THAT_ERROR(writeIFSToOutputStream(OS, Stub), Succeeded());
  compareByLine(Result.c_str(), Expected);
}
