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
    Ctx = std::make_unique<MCContext>(TT, MAI.get(), MRI.get(),
                                      /*MSTI=*/nullptr);
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
