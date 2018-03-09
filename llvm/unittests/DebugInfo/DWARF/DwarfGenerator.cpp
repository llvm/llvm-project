//===--- unittests/DebugInfo/DWARF/DwarfGenerator.cpp -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DwarfGenerator.h"
#include "../lib/CodeGen/AsmPrinter/DwarfStringPool.h"
#include "llvm/ADT/Triple.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.def"
#include "llvm/PassAnalysisSupport.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;
using namespace dwarf;

namespace {} // end anonymous namespace

//===----------------------------------------------------------------------===//
/// dwarfgen::DIE implementation.
//===----------------------------------------------------------------------===//
unsigned dwarfgen::DIE::computeSizeAndOffsets(unsigned Offset) {
  auto &DG = CU->getGenerator();
  return Die->computeOffsetsAndAbbrevs(DG.getAsmPrinter(), DG.getAbbrevSet(),
                                       Offset);
}

void dwarfgen::DIE::addAttribute(uint16_t A, dwarf::Form Form, uint64_t U) {
  auto &DG = CU->getGenerator();
  Die->addValue(DG.getAllocator(), static_cast<dwarf::Attribute>(A), Form,
                DIEInteger(U));
}

void dwarfgen::DIE::addAttribute(uint16_t A, dwarf::Form Form,
                                 StringRef String) {
  auto &DG = CU->getGenerator();
  if (Form == DW_FORM_string) {
    Die->addValue(DG.getAllocator(), static_cast<dwarf::Attribute>(A), Form,
                  new (DG.getAllocator())
                      DIEInlineString(String, DG.getAllocator()));
  } else {
    Die->addValue(
        DG.getAllocator(), static_cast<dwarf::Attribute>(A), Form,
        DIEString(DG.getStringPool().getEntry(*DG.getAsmPrinter(), String)));
  }
}

void dwarfgen::DIE::addAttribute(uint16_t A, dwarf::Form Form,
                                 dwarfgen::DIE &RefDie) {
  auto &DG = CU->getGenerator();
  Die->addValue(DG.getAllocator(), static_cast<dwarf::Attribute>(A), Form,
                DIEEntry(*RefDie.Die));
}

void dwarfgen::DIE::addAttribute(uint16_t A, dwarf::Form Form, const void *P,
                                 size_t S) {
  auto &DG = CU->getGenerator();
  DIEBlock *Block = new (DG.getAllocator()) DIEBlock;
  for (size_t I = 0; I < S; ++I)
    Block->addValue(
        DG.getAllocator(), (dwarf::Attribute)0, dwarf::DW_FORM_data1,
        DIEInteger(
            (const_cast<uint8_t *>(static_cast<const uint8_t *>(P)))[I]));

  Block->ComputeSize(DG.getAsmPrinter());
  Die->addValue(DG.getAllocator(), static_cast<dwarf::Attribute>(A), Form,
                Block);
}

void dwarfgen::DIE::addAttribute(uint16_t A, dwarf::Form Form) {
  auto &DG = CU->getGenerator();
  assert(Form == DW_FORM_flag_present);
  Die->addValue(DG.getAllocator(), static_cast<dwarf::Attribute>(A), Form,
                DIEInteger(1));
}

dwarfgen::DIE dwarfgen::DIE::addChild(dwarf::Tag Tag) {
  auto &DG = CU->getGenerator();
  return dwarfgen::DIE(CU,
                       &Die->addChild(llvm::DIE::get(DG.getAllocator(), Tag)));
}

dwarfgen::DIE dwarfgen::CompileUnit::getUnitDIE() {
  return dwarfgen::DIE(this, &DU.getUnitDie());
}

//===----------------------------------------------------------------------===//
/// dwarfgen::Generator implementation.
//===----------------------------------------------------------------------===//

dwarfgen::Generator::Generator()
    : MAB(nullptr), MCE(nullptr), MS(nullptr), StringPool(nullptr),
      Abbreviations(Allocator) {}
dwarfgen::Generator::~Generator() = default;

llvm::Expected<std::unique_ptr<dwarfgen::Generator>>
dwarfgen::Generator::create(Triple TheTriple, uint16_t DwarfVersion) {
  std::unique_ptr<dwarfgen::Generator> GenUP(new dwarfgen::Generator());
  llvm::Error error = GenUP->init(TheTriple, DwarfVersion);
  if (error)
    return Expected<std::unique_ptr<dwarfgen::Generator>>(std::move(error));
  return Expected<std::unique_ptr<dwarfgen::Generator>>(std::move(GenUP));
}

llvm::Error dwarfgen::Generator::init(Triple TheTriple, uint16_t V) {
  Version = V;
  std::string ErrorStr;
  std::string TripleName;

  // Get the target.
  const Target *TheTarget =
      TargetRegistry::lookupTarget(TripleName, TheTriple, ErrorStr);
  if (!TheTarget)
    return make_error<StringError>(ErrorStr, inconvertibleErrorCode());

  TripleName = TheTriple.getTriple();

  // Create all the MC Objects.
  MRI.reset(TheTarget->createMCRegInfo(TripleName));
  if (!MRI)
    return make_error<StringError>(Twine("no register info for target ") +
                                       TripleName,
                                   inconvertibleErrorCode());

  MAI.reset(TheTarget->createMCAsmInfo(*MRI, TripleName));
  if (!MAI)
    return make_error<StringError>("no asm info for target " + TripleName,
                                   inconvertibleErrorCode());

  MOFI.reset(new MCObjectFileInfo);
  MC.reset(new MCContext(MAI.get(), MRI.get(), MOFI.get()));
  MOFI->InitMCObjectFileInfo(TheTriple, /*PIC*/ false, *MC);

  MSTI.reset(TheTarget->createMCSubtargetInfo(TripleName, "", ""));
  if (!MSTI)
    return make_error<StringError>("no subtarget info for target " + TripleName,
                                   inconvertibleErrorCode());

  MCTargetOptions MCOptions = InitMCTargetOptionsFromFlags();
  MAB = TheTarget->createMCAsmBackend(*MSTI, *MRI, MCOptions);
  if (!MAB)
    return make_error<StringError>("no asm backend for target " + TripleName,
                                   inconvertibleErrorCode());

  MII.reset(TheTarget->createMCInstrInfo());
  if (!MII)
    return make_error<StringError>("no instr info info for target " +
                                       TripleName,
                                   inconvertibleErrorCode());

  MCE = TheTarget->createMCCodeEmitter(*MII, *MRI, *MC);
  if (!MCE)
    return make_error<StringError>("no code emitter for target " + TripleName,
                                   inconvertibleErrorCode());

  Stream = make_unique<raw_svector_ostream>(FileBytes);

  MS = TheTarget->createMCObjectStreamer(
      TheTriple, *MC, std::unique_ptr<MCAsmBackend>(MAB), *Stream,
      std::unique_ptr<MCCodeEmitter>(MCE), *MSTI, MCOptions.MCRelaxAll,
      MCOptions.MCIncrementalLinkerCompatible,
      /*DWARFMustBeAtTheEnd*/ false);
  if (!MS)
    return make_error<StringError>("no object streamer for target " +
                                       TripleName,
                                   inconvertibleErrorCode());

  // Finally create the AsmPrinter we'll use to emit the DIEs.
  TM.reset(TheTarget->createTargetMachine(TripleName, "", "", TargetOptions(),
                                          None));
  if (!TM)
    return make_error<StringError>("no target machine for target " + TripleName,
                                   inconvertibleErrorCode());

  Asm.reset(TheTarget->createAsmPrinter(*TM, std::unique_ptr<MCStreamer>(MS)));
  if (!Asm)
    return make_error<StringError>("no asm printer for target " + TripleName,
                                   inconvertibleErrorCode());

  // Set the DWARF version correctly on all classes that we use.
  MC->setDwarfVersion(Version);
  Asm->setDwarfVersion(Version);

  StringPool = llvm::make_unique<DwarfStringPool>(Allocator, *Asm, StringRef());

  return Error::success();
}

StringRef dwarfgen::Generator::generate() {
  // Offset from the first CU in the debug info section is 0 initially.
  unsigned SecOffset = 0;

  // Iterate over each compile unit and set the size and offsets for each
  // DIE within each compile unit. All offsets are CU relative.
  for (auto &CU : CompileUnits) {
    // Set the absolute .debug_info offset for this compile unit.
    CU->setOffset(SecOffset);
    // The DIEs contain compile unit relative offsets.
    unsigned CUOffset = 11;
    CUOffset = CU->getUnitDIE().computeSizeAndOffsets(CUOffset);
    // Update our absolute .debug_info offset.
    SecOffset += CUOffset;
    CU->setLength(CUOffset - 4);
  }
  Abbreviations.Emit(Asm.get(), MOFI->getDwarfAbbrevSection());
  StringPool->emit(*Asm, MOFI->getDwarfStrSection());
  MS->SwitchSection(MOFI->getDwarfInfoSection());
  for (auto &CU : CompileUnits) {
    uint16_t Version = CU->getVersion();
    auto Length = CU->getLength();
    MC->setDwarfVersion(Version);
    assert(Length != -1U);
    Asm->EmitInt32(Length);
    Asm->EmitInt16(Version);
    if (Version <= 4) {
      Asm->EmitInt32(0);
      Asm->EmitInt8(CU->getAddressSize());
    } else {
      Asm->EmitInt8(dwarf::DW_UT_compile);
      Asm->EmitInt8(CU->getAddressSize());
      Asm->EmitInt32(0);
    }
    Asm->emitDwarfDIE(*CU->getUnitDIE().Die);
  }

  MS->Finish();
  if (FileBytes.empty())
    return StringRef();
  return StringRef(FileBytes.data(), FileBytes.size());
}

bool dwarfgen::Generator::saveFile(StringRef Path) {
  if (FileBytes.empty())
    return false;
  std::error_code EC;
  raw_fd_ostream Strm(Path, EC, sys::fs::F_None);
  if (EC)
    return false;
  Strm.write(FileBytes.data(), FileBytes.size());
  Strm.close();
  return true;
}

dwarfgen::CompileUnit &dwarfgen::Generator::addCompileUnit() {
  CompileUnits.push_back(std::unique_ptr<CompileUnit>(
      new CompileUnit(*this, Version, Asm->getPointerSize())));
  return *CompileUnits.back();
}
