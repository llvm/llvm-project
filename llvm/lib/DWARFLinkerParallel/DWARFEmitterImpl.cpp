//===- DWARFEmitterImpl.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFEmitterImpl.h"
#include "llvm/DWARFLinker/DWARFLinkerCompileUnit.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FormattedStream.h"

namespace llvm {
namespace dwarflinker_parallel {

Error DwarfEmitterImpl::init(Triple TheTriple,
                             StringRef Swift5ReflectionSegmentName) {
  std::string ErrorStr;
  std::string TripleName;

  // Get the target.
  const Target *TheTarget =
      TargetRegistry::lookupTarget(TripleName, TheTriple, ErrorStr);
  if (!TheTarget)
    return createStringError(std::errc::invalid_argument, ErrorStr.c_str());
  TripleName = TheTriple.getTriple();

  // Create all the MC Objects.
  MRI.reset(TheTarget->createMCRegInfo(TripleName));
  if (!MRI)
    return createStringError(std::errc::invalid_argument,
                             "no register info for target %s",
                             TripleName.c_str());

  MCTargetOptions MCOptions = mc::InitMCTargetOptionsFromFlags();
  MAI.reset(TheTarget->createMCAsmInfo(*MRI, TripleName, MCOptions));
  if (!MAI)
    return createStringError(std::errc::invalid_argument,
                             "no asm info for target %s", TripleName.c_str());

  MSTI.reset(TheTarget->createMCSubtargetInfo(TripleName, "", ""));
  if (!MSTI)
    return createStringError(std::errc::invalid_argument,
                             "no subtarget info for target %s",
                             TripleName.c_str());

  MC.reset(new MCContext(TheTriple, MAI.get(), MRI.get(), MSTI.get(), nullptr,
                         nullptr, true, Swift5ReflectionSegmentName));
  MOFI.reset(TheTarget->createMCObjectFileInfo(*MC, /*PIC=*/false, false));
  MC->setObjectFileInfo(MOFI.get());

  MAB = TheTarget->createMCAsmBackend(*MSTI, *MRI, MCOptions);
  if (!MAB)
    return createStringError(std::errc::invalid_argument,
                             "no asm backend for target %s",
                             TripleName.c_str());

  MII.reset(TheTarget->createMCInstrInfo());
  if (!MII)
    return createStringError(std::errc::invalid_argument,
                             "no instr info info for target %s",
                             TripleName.c_str());

  MCE = TheTarget->createMCCodeEmitter(*MII, *MC);
  if (!MCE)
    return createStringError(std::errc::invalid_argument,
                             "no code emitter for target %s",
                             TripleName.c_str());

  switch (OutFileType) {
  case DWARFLinker::OutputFileType::Assembly: {
    MIP = TheTarget->createMCInstPrinter(TheTriple, MAI->getAssemblerDialect(),
                                         *MAI, *MII, *MRI);
    MS = TheTarget->createAsmStreamer(
        *MC, std::make_unique<formatted_raw_ostream>(OutFile), true, true, MIP,
        std::unique_ptr<MCCodeEmitter>(MCE), std::unique_ptr<MCAsmBackend>(MAB),
        true);
    break;
  }
  case DWARFLinker::OutputFileType::Object: {
    MS = TheTarget->createMCObjectStreamer(
        TheTriple, *MC, std::unique_ptr<MCAsmBackend>(MAB),
        MAB->createObjectWriter(OutFile), std::unique_ptr<MCCodeEmitter>(MCE),
        *MSTI, MCOptions.MCRelaxAll, MCOptions.MCIncrementalLinkerCompatible,
        /*DWARFMustBeAtTheEnd*/ false);
    break;
  }
  }

  if (!MS)
    return createStringError(std::errc::invalid_argument,
                             "no object streamer for target %s",
                             TripleName.c_str());

  // Finally create the AsmPrinter we'll use to emit the DIEs.
  TM.reset(TheTarget->createTargetMachine(TripleName, "", "", TargetOptions(),
                                          std::nullopt));
  if (!TM)
    return createStringError(std::errc::invalid_argument,
                             "no target machine for target %s",
                             TripleName.c_str());

  Asm.reset(TheTarget->createAsmPrinter(*TM, std::unique_ptr<MCStreamer>(MS)));
  if (!Asm)
    return createStringError(std::errc::invalid_argument,
                             "no asm printer for target %s",
                             TripleName.c_str());
  Asm->setDwarfUsesRelocationsAcrossSections(false);

  DebugInfoSectionSize = 0;

  return Error::success();
}

void DwarfEmitterImpl::emitSwiftAST(StringRef Buffer) {
  MCSection *SwiftASTSection = MOFI->getDwarfSwiftASTSection();
  SwiftASTSection->setAlignment(Align(32));
  MS->switchSection(SwiftASTSection);
  MS->emitBytes(Buffer);
}

/// Emit the swift reflection section stored in \p Buffer.
void DwarfEmitterImpl::emitSwiftReflectionSection(
    llvm::binaryformat::Swift5ReflectionSectionKind ReflSectionKind,
    StringRef Buffer, uint32_t Alignment, uint32_t) {
  MCSection *ReflectionSection =
      MOFI->getSwift5ReflectionSection(ReflSectionKind);
  if (ReflectionSection == nullptr)
    return;
  ReflectionSection->setAlignment(Align(Alignment));
  MS->switchSection(ReflectionSection);
  MS->emitBytes(Buffer);
}

void DwarfEmitterImpl::emitSectionContents(StringRef SecData,
                                           StringRef SecName) {
  if (SecData.empty())
    return;

  if (MCSection *Section = switchSection(SecName)) {
    MS->switchSection(Section);

    MS->emitBytes(SecData);
  }
}

MCSection *DwarfEmitterImpl::switchSection(StringRef SecName) {
  return StringSwitch<MCSection *>(SecName)
      .Case("debug_info", MC->getObjectFileInfo()->getDwarfInfoSection())
      .Case("debug_abbrev", MC->getObjectFileInfo()->getDwarfAbbrevSection())
      .Case("debug_line", MC->getObjectFileInfo()->getDwarfLineSection())
      .Case("debug_loc", MC->getObjectFileInfo()->getDwarfLocSection())
      .Case("debug_ranges", MC->getObjectFileInfo()->getDwarfRangesSection())
      .Case("debug_frame", MC->getObjectFileInfo()->getDwarfFrameSection())
      .Case("debug_aranges", MC->getObjectFileInfo()->getDwarfARangesSection())
      .Case("debug_rnglists",
            MC->getObjectFileInfo()->getDwarfRnglistsSection())
      .Case("debug_loclists",
            MC->getObjectFileInfo()->getDwarfLoclistsSection())
      .Case("debug_macro", MC->getObjectFileInfo()->getDwarfMacroSection())
      .Case("debug_macinfo", MC->getObjectFileInfo()->getDwarfMacinfoSection())
      .Case("debug_addr", MC->getObjectFileInfo()->getDwarfAddrSection())
      .Case("debug_str", MC->getObjectFileInfo()->getDwarfStrSection())
      .Case("debug_line_str", MC->getObjectFileInfo()->getDwarfLineStrSection())
      .Case("debug_str_offsets",
            MC->getObjectFileInfo()->getDwarfStrOffSection())
      .Default(nullptr);
}

MCSymbol *DwarfEmitterImpl::emitTempSym(StringRef SecName, StringRef SymName) {
  if (MCSection *Section = switchSection(SecName)) {
    MS->switchSection(Section);
    MCSymbol *Res = Asm->createTempSymbol(SymName);
    Asm->OutStreamer->emitLabel(Res);
    return Res;
  }

  return nullptr;
}

void DwarfEmitterImpl::emitAbbrevs(
    const SmallVector<std::unique_ptr<DIEAbbrev>> &Abbrevs,
    unsigned DwarfVersion) {
  MS->switchSection(MOFI->getDwarfAbbrevSection());
  MC->setDwarfVersion(DwarfVersion);
  Asm->emitDwarfAbbrevs(Abbrevs);
}

void DwarfEmitterImpl::emitCompileUnitHeader(DwarfUnit &Unit) {
  MS->switchSection(MOFI->getDwarfInfoSection());
  MC->setDwarfVersion(Unit.getVersion());

  // Emit size of content not including length itself. The size has already
  // been computed in CompileUnit::computeOffsets(). Subtract 4 to that size to
  // account for the length field.
  Asm->emitInt32(Unit.getDebugInfoHeaderSize() +
                 Unit.getOutUnitDIE()->getSize() - 4);
  Asm->emitInt16(Unit.getVersion());

  if (Unit.getVersion() >= 5) {
    Asm->emitInt8(dwarf::DW_UT_compile);
    Asm->emitInt8(Unit.getFormParams().AddrSize);
    // Proper offset to the abbreviations table will be set later.
    Asm->emitInt32(0);
    DebugInfoSectionSize += 12;
  } else {
    // Proper offset to the abbreviations table will be set later.
    Asm->emitInt32(0);
    Asm->emitInt8(Unit.getFormParams().AddrSize);
    DebugInfoSectionSize += 11;
  }
}

void DwarfEmitterImpl::emitDIE(DIE &Die) {
  MS->switchSection(MOFI->getDwarfInfoSection());
  Asm->emitDwarfDIE(Die);
  DebugInfoSectionSize += Die.getSize();
}

} // end of namespace dwarflinker_parallel
} // namespace llvm
