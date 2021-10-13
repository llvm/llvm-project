//===- llvm/unittest/MC/DwarfLineTableHeaders.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class DwarfLineTableHeaders : public ::testing::Test {
public:
  const char *TripleName = "x86_64-pc-linux";
  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MCAsmInfo> MAI;
  std::unique_ptr<const MCSubtargetInfo> STI;
  const Target *TheTarget;

  struct StreamerContext {
    std::unique_ptr<MCObjectFileInfo> MOFI;
    std::unique_ptr<MCContext> Ctx;
    std::unique_ptr<const MCInstrInfo> MII;
    std::unique_ptr<MCStreamer> Streamer;
  };

  DwarfLineTableHeaders() {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllDisassemblers();

    // If we didn't build x86, do not run the test.
    std::string Error;
    TheTarget = TargetRegistry::lookupTarget(TripleName, Error);
    if (!TheTarget)
      return;

    MRI.reset(TheTarget->createMCRegInfo(TripleName));
    MCTargetOptions MCOptions;
    MAI.reset(TheTarget->createMCAsmInfo(*MRI, TripleName, MCOptions));
    STI.reset(TheTarget->createMCSubtargetInfo(TripleName, "", ""));
  }

  /// Create all data structures necessary to operate an assembler
  StreamerContext createStreamer(raw_pwrite_stream &OS) {
    StreamerContext Res;
    Res.Ctx =
        std::make_unique<MCContext>(Triple(TripleName), MAI.get(), MRI.get(),
                                    /*MSTI=*/nullptr);
    Res.MOFI.reset(TheTarget->createMCObjectFileInfo(*Res.Ctx.get(),
                                                     /*PIC=*/false));
    Res.Ctx->setObjectFileInfo(Res.MOFI.get());

    Res.MII.reset(TheTarget->createMCInstrInfo());
    MCCodeEmitter *MCE =
        TheTarget->createMCCodeEmitter(*Res.MII, *MRI, *Res.Ctx);
    MCAsmBackend *MAB =
        TheTarget->createMCAsmBackend(*STI, *MRI, MCTargetOptions());
    std::unique_ptr<MCObjectWriter> OW = MAB->createObjectWriter(OS);
    Res.Streamer.reset(TheTarget->createMCObjectStreamer(
        Triple(TripleName), *Res.Ctx, std::unique_ptr<MCAsmBackend>(MAB),
        std::move(OW), std::unique_ptr<MCCodeEmitter>(MCE), *STI,
        /* RelaxAll */ false,
        /* IncrementalLinkerCompatible */ false,
        /* DWARFMustBeAtTheEnd */ false));
    return Res;
  }

  /// Emit a .debug_line section with the given context parameters
  void emitDebugLineSection(StreamerContext &C) {
    MCContext &Ctx = *C.Ctx;
    MCStreamer *TheStreamer = C.Streamer.get();
    MCAssembler &Assembler =
        static_cast<MCObjectStreamer *>(TheStreamer)->getAssembler();
    TheStreamer->initSections(false, *STI);

    // Create a mock function
    MCSection *Section = C.MOFI->getTextSection();
    Section->setHasInstructions(true);
    TheStreamer->SwitchSection(Section);
    TheStreamer->emitCFIStartProc(true);

    // Create a mock dwarfloc
    Ctx.setCurrentDwarfLoc(/*FileNo=*/0, /*Line=*/1, /*Column=*/1, /*Flags=*/0,
                           /*Isa=*/0, /*Discriminator=*/0);
    MCDwarfLoc Loc = Ctx.getCurrentDwarfLoc();
    MCSymbol *LineSym = Ctx.createTempSymbol();
    // Set the value of the symbol to use for the MCDwarfLineEntry.
    TheStreamer->emitLabel(LineSym);
    TheStreamer->emitNops(4, 1, SMLoc(), *STI);
    TheStreamer->emitCFIEndProc();

    // Start emission of .debug_line
    TheStreamer->SwitchSection(C.MOFI->getDwarfLineSection());
    MCDwarfLineTableHeader Header;
    MCDwarfLineTableParams Params = Assembler.getDWARFLinetableParams();
    Optional<MCDwarfLineStr> LineStr(None);
    if (Ctx.getDwarfVersion() >= 5) {
      LineStr = MCDwarfLineStr(Ctx);
      Header.setRootFile("dir", "file", None, None);
    }
    MCSymbol *LineEndSym = Header.Emit(TheStreamer, Params, LineStr).second;

    // Put out the line tables.
    MCLineSection::MCDwarfLineEntryCollection LineEntries;
    MCDwarfLineEntry LineEntry(LineSym, Loc);
    LineEntries.push_back(LineEntry);
    MCDwarfLineTable::emitOne(TheStreamer, Section, LineEntries);
    TheStreamer->emitLabel(LineEndSym);
    if (LineStr)
      LineStr->emitSection(TheStreamer);
  }

  /// Check contents of .debug_line section
  void verifyDebugLineContents(const llvm::object::ObjectFile &E,
                               ArrayRef<uint8_t> ExpectedEncoding) {
    for (const llvm::object::SectionRef &Section : E.sections()) {
      Expected<StringRef> SectionNameOrErr = Section.getName();
      ASSERT_TRUE(static_cast<bool>(SectionNameOrErr));
      StringRef SectionName = *SectionNameOrErr;
      if (SectionName.empty() || SectionName != ".debug_line")
        continue;
      Expected<StringRef> ContentsOrErr = Section.getContents();
      ASSERT_TRUE(static_cast<bool>(ContentsOrErr));
      StringRef Contents = *ContentsOrErr;
      ASSERT_TRUE(Contents.size() > ExpectedEncoding.size());
      EXPECT_EQ(
          arrayRefFromStringRef(Contents.slice(0, ExpectedEncoding.size())),
          ExpectedEncoding);
      return;
    }
    llvm_unreachable(".debug_line not found");
  }

  ///  Open ObjFileData as an object file and read its .debug_line section
  void readAndCheckDebugLineContents(StringRef ObjFileData,
                                     ArrayRef<uint8_t> Expected) {
    std::unique_ptr<MemoryBuffer> MB =
        MemoryBuffer::getMemBuffer(ObjFileData, "", false);
    std::unique_ptr<object::Binary> Bin =
        cantFail(llvm::object::createBinary(MB->getMemBufferRef()));
    if (auto *E = dyn_cast<llvm::object::ELFObjectFileBase>(&*Bin)) {
      return verifyDebugLineContents(*E, Expected);
    }
    llvm_unreachable("ELF object file not found");
  }
};
} // namespace

TEST_F(DwarfLineTableHeaders, TestDWARF4HeaderEmission) {
  if (!MRI)
    return;

  SmallString<0> EmittedBinContents;
  raw_svector_ostream VecOS(EmittedBinContents);
  StreamerContext C = createStreamer(VecOS);
  C.Ctx->setDwarfVersion(4);
  emitDebugLineSection(C);
  C.Streamer->Finish();
  readAndCheckDebugLineContents(
      EmittedBinContents.str(),
      {/*    Total length=*/0x30, 0, 0, 0,
       /*   DWARF version=*/4, 0,
       /* Prologue length=*/0x14, 0, 0, 0,
       /* min_inst_length=*/1,
       /*max_ops_per_inst=*/1,
       /* default_is_stmt=*/DWARF2_LINE_DEFAULT_IS_STMT,
       /*       line_base=*/static_cast<uint8_t>(-5),
       /*      line_range=*/14,
       /*     opcode_base=*/13});
}

TEST_F(DwarfLineTableHeaders, TestDWARF5HeaderEmission) {
  if (!MRI)
    return;

  SmallString<0> EmittedBinContents;
  raw_svector_ostream VecOS(EmittedBinContents);
  StreamerContext C = createStreamer(VecOS);
  C.Ctx->setDwarfVersion(5);
  emitDebugLineSection(C);
  C.Streamer->Finish();
  readAndCheckDebugLineContents(
      EmittedBinContents.str(),
      {/*    Total length=*/0x43, 0, 0, 0,
       /*   DWARF version=*/5, 0,
       /*        ptr size=*/8,
       /*         segment=*/0,
       /* Prologue length=*/0x25, 0, 0, 0,
       /* min_inst_length=*/1,
       /*max_ops_per_inst=*/1,
       /* default_is_stmt=*/DWARF2_LINE_DEFAULT_IS_STMT,
       /*       line_base=*/static_cast<uint8_t>(-5),
       /*      line_range=*/14,
       /*     opcode_base=*/13});
}
