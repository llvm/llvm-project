//===- llvm/unittest/MC/DwarfLineTables.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
struct Context {
  static constexpr char TripleName[] = "x86_64-pc-linux";
  Triple TT;
  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MCAsmInfo> MAI;
  std::unique_ptr<MCSubtargetInfo> STI;
  std::unique_ptr<MCContext> Ctx;

  Context() : TT(TripleName) {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllDisassemblers();

    // If we didn't build x86, do not run the test.
    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
    if (!TheTarget)
      return;

    MRI.reset(TheTarget->createMCRegInfo(TT));
    MCTargetOptions MCOptions;
    MAI.reset(TheTarget->createMCAsmInfo(*MRI, TT, MCOptions));
    STI.reset(TheTarget->createMCSubtargetInfo(TT, "", ""));
    Ctx = std::make_unique<MCContext>(TT, *MAI, *MRI, *STI);
  }

  operator bool() { return Ctx.get(); }
  operator MCContext &() { return *Ctx; };
};

Context &getContext() {
  static Context Ctxt;
  return Ctxt;
}
}

void verifyEncoding(MCDwarfLineTableParams Params, int LineDelta, int AddrDelta,
                    ArrayRef<uint8_t> ExpectedEncoding) {
  SmallString<16> Buffer;
  MCDwarfLineAddr::encode(getContext(), Params, LineDelta, AddrDelta,
                          Buffer);
  EXPECT_EQ(ExpectedEncoding, arrayRefFromStringRef(Buffer));
}

TEST(DwarfLineTables, TestDefaultParams) {
  if (!getContext())
    GTEST_SKIP();

  MCDwarfLineTableParams Params;

  // Minimal line offset expressible through extended opcode, 0 addr delta
  const uint8_t Encoding0[] = {13}; // Special opcode Addr += 0, Line += -5
  verifyEncoding(Params, -5, 0, Encoding0);

  // Maximal line offset expressible through extended opcode,
  const uint8_t Encoding1[] = {26}; // Special opcode Addr += 0, Line += +8
  verifyEncoding(Params, 8, 0, Encoding1);

  // Random value in the middle of the special ocode range
  const uint8_t Encoding2[] = {146}; // Special opcode Addr += 9, Line += 2
  verifyEncoding(Params, 2, 9, Encoding2);

  // Minimal line offset expressible through extended opcode, max addr delta
  const uint8_t Encoding3[] = {251}; // Special opcode Addr += 17, Line += -5
  verifyEncoding(Params, -5, 17, Encoding3);

  // Biggest special opcode
  const uint8_t Encoding4[] = {255}; // Special opcode Addr += 17, Line += -1
  verifyEncoding(Params, -1, 17, Encoding4);

  // Line delta outside of the special opcode range, address delta in range
  const uint8_t Encoding5[] = {dwarf::DW_LNS_advance_line, 9,
                               158}; // Special opcode Addr += 10, Line += 0
  verifyEncoding(Params, 9, 10, Encoding5);

  // Address delta outside of the special opcode range, but small
  // enough to do DW_LNS_const_add_pc + special opcode.
  const uint8_t Encoding6[] = {dwarf::DW_LNS_const_add_pc, // pc += 17
                               62}; // Special opcode Addr += 3, Line += 2
  verifyEncoding(Params, 2, 20, Encoding6);

  // Address delta big enough to require the use of DW_LNS_advance_pc
  // Line delta in special opcode range
  const uint8_t Encoding7[] = {dwarf::DW_LNS_advance_pc, 100,
                               20}; // Special opcode Addr += 0, Line += 2
  verifyEncoding(Params, 2, 100, Encoding7);

  // No special opcode possible.
  const uint8_t Encoding8[] = {dwarf::DW_LNS_advance_line, 20,
                               dwarf::DW_LNS_advance_pc, 100,
                               dwarf::DW_LNS_copy};
  verifyEncoding(Params, 20, 100, Encoding8);
}

TEST(DwarfLineTables, TestCustomParams) {
  if (!getContext())
    GTEST_SKIP();

  // Some tests against the example values given in the standard.
  MCDwarfLineTableParams Params;
  Params.DWARF2LineOpcodeBase = 13;
  Params.DWARF2LineBase = -3;
  Params.DWARF2LineRange = 12;

  // Minimal line offset expressible through extended opcode, 0 addr delta
  const uint8_t Encoding0[] = {13}; // Special opcode Addr += 0, Line += -5
  verifyEncoding(Params, -3, 0, Encoding0);

  // Maximal line offset expressible through extended opcode,
  const uint8_t Encoding1[] = {24}; // Special opcode Addr += 0, Line += +8
  verifyEncoding(Params, 8, 0, Encoding1);

  // Random value in the middle of the special ocode range
  const uint8_t Encoding2[] = {126}; // Special opcode Addr += 9, Line += 2
  verifyEncoding(Params, 2, 9, Encoding2);

  // Minimal line offset expressible through extended opcode, max addr delta
  const uint8_t Encoding3[] = {253}; // Special opcode Addr += 20, Line += -3
  verifyEncoding(Params, -3, 20, Encoding3);

  // Biggest special opcode
  const uint8_t Encoding4[] = {255}; // Special opcode Addr += 17, Line += -1
  verifyEncoding(Params, -1, 20, Encoding4);

  // Line delta outside of the special opcode range, address delta in range
  const uint8_t Encoding5[] = {dwarf::DW_LNS_advance_line, 9,
                               136}; // Special opcode Addr += 10, Line += 0
  verifyEncoding(Params, 9, 10, Encoding5);

  // Address delta outside of the special opcode range, but small
  // enough to do DW_LNS_const_add_pc + special opcode.
  const uint8_t Encoding6[] = {dwarf::DW_LNS_const_add_pc, // pc += 20
                               138}; // Special opcode Addr += 10, Line += 2
  verifyEncoding(Params, 2, 30, Encoding6);

  // Address delta big enough to require the use of DW_LNS_advance_pc
  // Line delta in special opcode range
  const uint8_t Encoding7[] = {dwarf::DW_LNS_advance_pc, 100,
                               18}; // Special opcode Addr += 0, Line += 2
  verifyEncoding(Params, 2, 100, Encoding7);

  // No special opcode possible.
  const uint8_t Encoding8[] = {dwarf::DW_LNS_advance_line, 20,
                               dwarf::DW_LNS_advance_pc, 100,
                               dwarf::DW_LNS_copy};
  verifyEncoding(Params, 20, 100, Encoding8);
}

TEST(DwarfLineTables, TestCustomParams2) {
  if (!getContext())
    GTEST_SKIP();

  // Corner case param values.
  MCDwarfLineTableParams Params;
  Params.DWARF2LineOpcodeBase = 13;
  Params.DWARF2LineBase = 1;
  Params.DWARF2LineRange = 255;

  const uint8_t Encoding0[] = {dwarf::DW_LNS_advance_line, 248, 1,
                               dwarf::DW_LNS_copy};
  verifyEncoding(Params, 248, 0, Encoding0);
}


TEST(DwarfLineTables, RootFilePathIdentity) {
  struct Case {
    const char *Root;
    const char *Directory;
    const char *Name;
    unsigned Expected;
  };
  const Case Cases[] = {
      {"/build/src/a.c", "/build/src", "a.c", 0},
      {"/build/src/a.c", "src", "a.c", 0},
      {"/build/src/a.c", "", "src/a.c", 0},
      {"src/a.c", "", "/build/src/a.c", 0},
      {"a.c", "", "a.c", 0},
      {"a.c", "/build", "a.c", 0},
      {"a.c", "/other", "a.c", 1},
      {"a.c", "other", "a.c", 1},
      {"/build/a.c", "/ignored", "/build/a.c", 0},
      {"C:\\build\\a.c", "/ignored", "C:\\build\\a.c", 0},
  };
  for (const auto &C : Cases) {
    SCOPED_TRACE(C.Root);
    SCOPED_TRACE(C.Directory);
    SCOPED_TRACE(C.Name);
    MCDwarfLineTableHeader Header;
    Header.setRootFile("/build", C.Root, std::nullopt, "source text");
    StringRef Directory = C.Directory, Name = C.Name;
    EXPECT_EQ(C.Expected, cantFail(Header.tryGetFile(
                              Directory, Name, std::nullopt, "source text", 5)));
    // Root spelling and embedded source must survive deduplication.
    EXPECT_EQ(C.Root, Header.RootFile.Name);
    EXPECT_EQ("source text", *Header.RootFile.Source);
  }
}

TEST(DwarfLineTables, RootFileForeignPathStyles) {
  MCDwarfLineTableHeader Header;
  Header.setRootFile("C:\\build", "C:/build/src/a.c", std::nullopt, std::nullopt);
  StringRef Directory = "src", Name = "a.c";
  EXPECT_EQ(0u, cantFail(Header.tryGetFile(Directory, Name, std::nullopt,
                                         std::nullopt, 5)));
  Directory = "C:\\other\\src";
  Name = "a.c";
  EXPECT_EQ(1u, cantFail(Header.tryGetFile(Directory, Name, std::nullopt,
                                         std::nullopt, 5)));

  // A backslash in a POSIX path is a character, not a separator.
  MCDwarfLineTableHeader Posix;
  Posix.setRootFile("/build", "/build/a\\b.c", std::nullopt, std::nullopt);
  Directory = "/build/a";
  Name = "b.c";
  EXPECT_EQ(1u, cantFail(Posix.tryGetFile(Directory, Name, std::nullopt,
                                        std::nullopt, 5)));
}

TEST(DwarfLineTables, RootFileChecksumIdentity) {
  MD5::MD5Result First{}, Second{};
  Second[0] = 1;
  for (std::optional<MD5::MD5Result> Checksum :
       {std::optional<MD5::MD5Result>(First),
        std::optional<MD5::MD5Result>(Second), std::optional<MD5::MD5Result>()}) {
    MCDwarfLineTableHeader Header;
    Header.setRootFile("/build", "/build/a.c", First, std::nullopt);
    StringRef Directory = "/build", Name = "a.c";
    EXPECT_EQ(Checksum == First ? 0u : 1u,
              cantFail(Header.tryGetFile(Directory, Name, Checksum,
                                         std::nullopt, 5)));
  }
}

TEST(DwarfLineTables, ExplicitRootFileAliasKeepsItsNumber) {
  MCDwarfLineTableHeader Header;
  Header.setRootFile("/build", "a.c", std::nullopt, std::nullopt);
  StringRef Directory = "/build", Name = "a.c";
  EXPECT_EQ(1u, cantFail(Header.tryGetFile(Directory, Name, std::nullopt,
                                          std::nullopt, 5, 1)));
  ASSERT_EQ(2u, Header.MCDwarfFiles.size());
  EXPECT_EQ("a.c", Header.MCDwarfFiles[1].Name);
}

TEST(DwarfLineTables, DWARF4DoesNotReuseFileZero) {
  MCDwarfLineTableHeader Header;
  Header.setRootFile("/build", "/build/a.c", std::nullopt, std::nullopt);
  StringRef Directory = "/build", Name = "a.c";
  EXPECT_EQ(1u, cantFail(Header.tryGetFile(Directory, Name, std::nullopt,
                                          std::nullopt, 4)));
}
