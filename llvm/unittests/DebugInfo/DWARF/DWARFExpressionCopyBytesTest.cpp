//===- llvm/unittest/DebugInfo/DWARFExpressionRawDataTest.cpp -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFExpression.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace dwarf;

namespace {

/// Tests that a client of DebugInfo/DWARF is able to read raw data bytes of a
/// DWARFExpression parsed from CFI with the intent of writing them back as is
/// via MC layer / cfi_escape.
/// This is relevant for binary tools that need to rewrite/copy unwind and
/// debug info from input to output binary.
class DWARFExpressionCopyBytesTest : public ::testing::Test {
public:
  const char *TripleName = "x86_64-pc-linux";
  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MCAsmInfo> MAI;
  std::unique_ptr<const MCSubtargetInfo> STI;
  const Target *TheTarget;

  DWARFExpressionCopyBytesTest() {
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmPrinters();

    std::string ErrorStr;
    TheTarget = TargetRegistry::lookupTarget(TripleName, ErrorStr);
    if (!TheTarget)
      return;

    MRI.reset(TheTarget->createMCRegInfo(TripleName));
    MAI.reset(TheTarget->createMCAsmInfo(*MRI, TripleName, MCTargetOptions()));
    STI.reset(TheTarget->createMCSubtargetInfo(TripleName, "", ""));
  }

  struct StreamerContext {
    std::unique_ptr<MCObjectFileInfo> MOFI;
    std::unique_ptr<MCContext> Ctx;
    std::unique_ptr<const MCInstrInfo> MII;
    std::unique_ptr<MCStreamer> Streamer;
  };

  /// Create all data structures necessary to operate an assembler
  StreamerContext createStreamer(raw_pwrite_stream &OS);
  /// Emit a dummy obj file with a single CFI instruction,
  /// DW_CFA_def_cfa_expression, encoding as its operand the DWARF expression
  /// represented by ExprBytes
  SmallString<0> emitObjFile(StringRef ExprBytes);
  /// Peruse the object file looking for the encoded DWARF expression, and check
  /// that its operand was encoded correctly
  void parseCFIsAndCheckExpression(const llvm::object::ObjectFile &E,
                                   ArrayRef<uint8_t> Expected);
  /// Open the in-memory relocatable object file and verify that it contains
  /// the expected DWARF expression bytes
  void readAndCheckObjFile(StringRef ObjFileData, ArrayRef<uint8_t> Expected);
  /// Run this test on the DWARF expression represented by the bytes in
  /// ExprData. Check that the getData() API retrieves these original bytes and
  /// that we can use them to encode a CFI with those bytes as operands (via
  /// cfi_escape).
  void testExpr(ArrayRef<uint8_t> ExprData);
};

} // namespace

DWARFExpressionCopyBytesTest::StreamerContext
DWARFExpressionCopyBytesTest::createStreamer(raw_pwrite_stream &OS) {
  StreamerContext Res;
  Res.Ctx =
      std::make_unique<MCContext>(Triple(TripleName), MAI.get(), MRI.get(),
                                  /*MSTI=*/nullptr);
  Res.MOFI.reset(TheTarget->createMCObjectFileInfo(*Res.Ctx,
                                                   /*PIC=*/false));
  Res.Ctx->setObjectFileInfo(Res.MOFI.get());

  Res.MII.reset(TheTarget->createMCInstrInfo());
  MCCodeEmitter *MCE = TheTarget->createMCCodeEmitter(*Res.MII, *Res.Ctx);
  MCAsmBackend *MAB =
      TheTarget->createMCAsmBackend(*STI, *MRI, MCTargetOptions());
  std::unique_ptr<MCObjectWriter> OW = MAB->createObjectWriter(OS);
  Res.Streamer.reset(TheTarget->createMCObjectStreamer(
      Triple(TripleName), *Res.Ctx, std::unique_ptr<MCAsmBackend>(MAB),
      std::move(OW), std::unique_ptr<MCCodeEmitter>(MCE), *STI));
  return Res;
}

SmallString<0> DWARFExpressionCopyBytesTest::emitObjFile(StringRef ExprBytes) {
  auto EncodeDefCfaExpr = [&](StringRef Bytes) {
    std::string Str;
    raw_string_ostream OS(Str);
    OS << static_cast<uint8_t>(dwarf::DW_CFA_def_cfa_expression);
    encodeULEB128(Bytes.size(), OS);
    OS << Bytes;
    return Str;
  };

  SmallString<0> Storage;
  raw_svector_ostream VecOS(Storage);
  StreamerContext C = createStreamer(VecOS);
  C.Streamer->initSections(false, *STI);
  MCSection *Section = C.MOFI->getTextSection();
  Section->setHasInstructions(true);
  C.Streamer->switchSection(Section);
  C.Streamer->emitCFIStartProc(true);
  auto Str = EncodeDefCfaExpr(ExprBytes);
  C.Streamer->emitCFIEscape(Str);
  C.Streamer->emitNops(4, 1, SMLoc(), *STI);
  C.Streamer->emitCFIEndProc();
  C.Streamer->finish();
  return Storage;
}

void DWARFExpressionCopyBytesTest::parseCFIsAndCheckExpression(
    const llvm::object::ObjectFile &E, ArrayRef<uint8_t> Expected) {
  auto FetchFirstCfaExpression = [](const DWARFDebugFrame &EHFrame)
      -> std::optional<CFIProgram::Instruction> {
    for (const dwarf::FrameEntry &Entry : EHFrame.entries()) {
      const auto *CurFDE = dyn_cast<dwarf::FDE>(&Entry);
      if (!CurFDE)
        continue;
      for (const CFIProgram::Instruction &Instr : CurFDE->cfis()) {
        if (Instr.Opcode != dwarf::DW_CFA_def_cfa_expression)
          continue;
        return Instr;
      }
    }
    return std::nullopt;
  };

  std::unique_ptr<DWARFContext> Ctx = DWARFContext::create(E);
  const DWARFDebugFrame *EHFrame = cantFail(Ctx->getEHFrame());
  ASSERT_NE(nullptr, EHFrame);
  auto CfiInstr = FetchFirstCfaExpression(*EHFrame);
  ASSERT_TRUE(CfiInstr);
  DWARFExpression Expr = *(CfiInstr->Expression);
  StringRef ExprData = Expr.getData();
  EXPECT_EQ(ExprData.size(), Expected.size());
  for (unsigned I = 0, E = ExprData.size(); I != E; ++I) {
    EXPECT_EQ(static_cast<uint8_t>(ExprData[I]), Expected[I]);
  }
}

void DWARFExpressionCopyBytesTest::readAndCheckObjFile(
    StringRef ObjFileData, ArrayRef<uint8_t> Expected) {
  std::unique_ptr<MemoryBuffer> MB =
      MemoryBuffer::getMemBuffer(ObjFileData, "", false);
  std::unique_ptr<object::Binary> Bin =
      cantFail(llvm::object::createBinary(MB->getMemBufferRef()));
  if (auto *E = dyn_cast<llvm::object::ELFObjectFileBase>(&*Bin)) {
    parseCFIsAndCheckExpression(*E, Expected);
  }
}

void DWARFExpressionCopyBytesTest::testExpr(ArrayRef<uint8_t> ExprData) {
  // If we didn't build x86, do not run the test.
  if (!MRI)
    GTEST_SKIP();

  DataExtractor DE(ExprData, true, 8);
  DWARFExpression Expr(DE, 8);

  // Copy this expression into the CFI of a binary and check that we are able to
  // get it back correctly from this binary.
  const SmallString<0> EmittedBinContents = emitObjFile(Expr.getData());
  readAndCheckObjFile(EmittedBinContents.str(), ExprData);
}

TEST_F(DWARFExpressionCopyBytesTest, Test_OP_reg0) { testExpr({DW_OP_reg0}); }

TEST_F(DWARFExpressionCopyBytesTest, Test_OP_reg10) { testExpr({DW_OP_reg10}); }

TEST_F(DWARFExpressionCopyBytesTest, Test_OP_regx) {
  testExpr({DW_OP_regx, 0x80, 0x02});
}

TEST_F(DWARFExpressionCopyBytesTest, Test_OP_breg0) {
  testExpr({DW_OP_breg0, 0x04});
}

TEST_F(DWARFExpressionCopyBytesTest, Test_OP_breg0_large_offset) {
  testExpr({DW_OP_breg0, 0x80, 0x02});
}

TEST_F(DWARFExpressionCopyBytesTest, Test_OP_breg13) {
  testExpr({DW_OP_breg13, 0x10});
}

TEST_F(DWARFExpressionCopyBytesTest, Test_OP_breg13_zero_offset) {
  testExpr({DW_OP_breg13, 0x00});
}

TEST_F(DWARFExpressionCopyBytesTest, Test_OP_breg0_negative) {
  testExpr({DW_OP_breg0, 0x70});
}

TEST_F(DWARFExpressionCopyBytesTest, Test_OP_bregx) {
  testExpr({DW_OP_bregx, 0x0d, 0x28});
}

TEST_F(DWARFExpressionCopyBytesTest, Test_OP_stack_value) {
  testExpr({DW_OP_breg13, 0x04, DW_OP_stack_value});
}

TEST_F(DWARFExpressionCopyBytesTest, Test_OP_entry_value) {
  testExpr({DW_OP_entry_value, 0x01, DW_OP_reg0, DW_OP_stack_value});
}

TEST_F(DWARFExpressionCopyBytesTest, Test_OP_entry_value_mem) {
  testExpr({DW_OP_entry_value, 0x02, DW_OP_breg13, 0x10, DW_OP_stack_value});
}
