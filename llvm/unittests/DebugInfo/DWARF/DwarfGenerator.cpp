//===--- unittests/DebugInfo/DWARF/DwarfGenerator.cpp -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DwarfGenerator.h"
#include "../lib/CodeGen/AsmPrinter/DwarfStringPool.h"
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
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using namespace dwarf;

mc::RegisterMCTargetOptionsFlags MOF;

namespace {} // end anonymous namespace

//===----------------------------------------------------------------------===//
/// dwarfgen::DIE implementation.
//===----------------------------------------------------------------------===//
unsigned dwarfgen::DIE::computeSizeAndOffsets(unsigned Offset) {
  auto &DG = CU->getGenerator();
  return Die->computeOffsetsAndAbbrevs(DG.getAsmPrinter()->getDwarfFormParams(),
                                       DG.getAbbrevSet(), Offset);
}

void dwarfgen::DIE::addAttribute(uint16_t A, dwarf::Form Form, uint64_t U) {
  auto &DG = CU->getGenerator();
  switch (Form) {
  case DW_FORM_addrx:
  case DW_FORM_addrx1:
  case DW_FORM_addrx2:
  case DW_FORM_addrx3:
  case DW_FORM_addrx4:
    U = DG.getAddressPool().getIndex(U);
    break;
  default:
    break;
  }
  Die->addValue(DG.getAllocator(), static_cast<dwarf::Attribute>(A), Form,
                DIEInteger(U));
}

void dwarfgen::DIE::addAttribute(uint16_t A, dwarf::Form Form, const MCExpr &Expr) {
  auto &DG = CU->getGenerator();
  Die->addValue(DG.getAllocator(), static_cast<dwarf::Attribute>(A), Form,
                DIEExpr(&Expr));
}

void dwarfgen::DIE::addAttribute(uint16_t A, dwarf::Form Form,
                                 StringRef String) {
  auto &DG = CU->getGenerator();
  switch (Form) {
  case DW_FORM_string:
    Die->addValue(DG.getAllocator(), static_cast<dwarf::Attribute>(A), Form,
                  new (DG.getAllocator())
                      DIEInlineString(String, DG.getAllocator()));
    break;

  case DW_FORM_strp:
    Die->addValue(
        DG.getAllocator(), static_cast<dwarf::Attribute>(A), Form,
        DIEString(DG.getStringPool().getEntry(*DG.getAsmPrinter(), String)));
    break;

  case DW_FORM_GNU_str_index:
  case DW_FORM_strx:
  case DW_FORM_strx1:
  case DW_FORM_strx2:
  case DW_FORM_strx3:
  case DW_FORM_strx4:
    Die->addValue(DG.getAllocator(), static_cast<dwarf::Attribute>(A), Form,
                  DIEString(DG.getStringPool().getIndexedEntry(
                      *DG.getAsmPrinter(), String)));
    break;

  default:
    llvm_unreachable("Unhandled form!");
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

  Block->computeSize(DG.getAsmPrinter()->getDwarfFormParams());
  Die->addValue(DG.getAllocator(), static_cast<dwarf::Attribute>(A), Form,
                Block);
}

void dwarfgen::DIE::addAttribute(uint16_t A, dwarf::Form Form) {
  auto &DG = CU->getGenerator();
  assert(Form == DW_FORM_flag_present);
  Die->addValue(DG.getAllocator(), static_cast<dwarf::Attribute>(A), Form,
                DIEInteger(1));
}

void dwarfgen::DIE::addStrOffsetsBaseAttribute() {
  auto &DG = CU->getGenerator();
  auto &MC = *DG.getMCContext();
  AsmPrinter *Asm = DG.getAsmPrinter();

  const MCSymbol *SectionStart =
      Asm->getObjFileLowering().getDwarfStrOffSection()->getBeginSymbol();

  const MCExpr *Expr =
      MCSymbolRefExpr::create(DG.getStringOffsetsStartSym(), MC);

  if (!Asm->MAI->doesDwarfUseRelocationsAcrossSections())
    Expr = MCBinaryExpr::createSub(
        Expr, MCSymbolRefExpr::create(SectionStart, MC), MC);

  addAttribute(dwarf::DW_AT_str_offsets_base, DW_FORM_sec_offset, *Expr);
}

// This is currently fixed to be the first address entry after the header.
void dwarfgen::DIE::addAddrBaseAttribute() {
  auto &DG = CU->getGenerator();
  auto &MC = *DG.getMCContext();
  AsmPrinter *Asm = DG.getAsmPrinter();

  const MCSymbol *SectionStart =
      Asm->getObjFileLowering().getDwarfAddrSection()->getBeginSymbol();

  const MCExpr *Expr = MCSymbolRefExpr::create(DG.getAddrTableStartSym(), MC);

  if (!Asm->MAI->doesDwarfUseRelocationsAcrossSections())
    Expr = MCBinaryExpr::createSub(
        Expr, MCSymbolRefExpr::create(SectionStart, MC), MC);

  addAttribute(dwarf::DW_AT_addr_base, DW_FORM_sec_offset, *Expr);
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
/// dwarfgen::LineTable implementation.
//===----------------------------------------------------------------------===//
DWARFDebugLine::Prologue dwarfgen::LineTable::createBasicPrologue() const {
  DWARFDebugLine::Prologue P;
  switch (Version) {
  case 2:
  case 3:
    P.TotalLength = 41;
    P.PrologueLength = 35;
    break;
  case 4:
    P.TotalLength = 42;
    P.PrologueLength = 36;
    break;
  case 5:
    P.TotalLength = 50;
    P.PrologueLength = 42;
    P.FormParams.AddrSize = AddrSize;
    break;
  default:
    llvm_unreachable("unsupported version");
  }
  if (Format == DWARF64) {
    P.TotalLength += 4;
    P.FormParams.Format = DWARF64;
  }
  P.TotalLength += getContentsSize();
  P.FormParams.Version = Version;
  P.MinInstLength = 1;
  P.MaxOpsPerInst = 1;
  P.DefaultIsStmt = 1;
  P.LineBase = -5;
  P.LineRange = 14;
  P.OpcodeBase = 13;
  P.StandardOpcodeLengths = {0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1};
  P.IncludeDirectories.push_back(
      DWARFFormValue::createFromPValue(DW_FORM_string, "a dir"));
  P.FileNames.push_back(DWARFDebugLine::FileNameEntry());
  P.FileNames.back().Name =
      DWARFFormValue::createFromPValue(DW_FORM_string, "a file");
  return P;
}

void dwarfgen::LineTable::setPrologue(DWARFDebugLine::Prologue NewPrologue) {
  Prologue = NewPrologue;
  CustomPrologue.clear();
}

void dwarfgen::LineTable::setCustomPrologue(
    ArrayRef<ValueAndLength> NewPrologue) {
  Prologue.reset();
  CustomPrologue = NewPrologue;
}

void dwarfgen::LineTable::addByte(uint8_t Value) {
  Contents.push_back({Value, Byte});
}

void dwarfgen::LineTable::addStandardOpcode(uint8_t Opcode,
                                            ArrayRef<ValueAndLength> Operands) {
  Contents.push_back({Opcode, Byte});
  Contents.insert(Contents.end(), Operands.begin(), Operands.end());
}

void dwarfgen::LineTable::addExtendedOpcode(uint64_t Length, uint8_t Opcode,
                                            ArrayRef<ValueAndLength> Operands) {
  Contents.push_back({0, Byte});
  Contents.push_back({Length, ULEB});
  Contents.push_back({Opcode, Byte});
  Contents.insert(Contents.end(), Operands.begin(), Operands.end());
}

void dwarfgen::LineTable::generate(MCContext &MC, AsmPrinter &Asm) const {
  MC.setDwarfVersion(Version);

  MCSymbol *EndSymbol = nullptr;
  if (!CustomPrologue.empty()) {
    writeData(CustomPrologue, Asm);
  } else if (!Prologue) {
    EndSymbol = writeDefaultPrologue(Asm);
  } else {
    writePrologue(Asm);
  }

  writeData(Contents, Asm);
  if (EndSymbol != nullptr)
    Asm.OutStreamer->emitLabel(EndSymbol);
}

void dwarfgen::LineTable::writeData(ArrayRef<ValueAndLength> Data,
                                    AsmPrinter &Asm) const {
  for (auto Entry : Data) {
    switch (Entry.Length) {
    case Byte:
    case Half:
    case Long:
    case Quad:
      Asm.OutStreamer->emitIntValue(Entry.Value, Entry.Length);
      continue;
    case ULEB:
      Asm.emitULEB128(Entry.Value);
      continue;
    case SLEB:
      Asm.emitSLEB128(Entry.Value);
      continue;
    }
    llvm_unreachable("unsupported ValueAndLength Length value");
  }
}

size_t dwarfgen::LineTable::getContentsSize() const {
  size_t Size = 0;
  for (auto Entry : Contents) {
    switch (Entry.Length) {
    case ULEB:
      Size += getULEB128Size(Entry.Value);
      break;
    case SLEB:
      Size += getSLEB128Size(Entry.Value);
      break;
    default:
      Size += Entry.Length;
      break;
    }
  }
  return Size;
}

MCSymbol *dwarfgen::LineTable::writeDefaultPrologue(AsmPrinter &Asm) const {
  MCSymbol *UnitStart = Asm.createTempSymbol("line_unit_start");
  MCSymbol *UnitEnd = Asm.createTempSymbol("line_unit_end");
  if (Format == DwarfFormat::DWARF64) {
    Asm.emitInt32((int)dwarf::DW_LENGTH_DWARF64);
    Asm.emitLabelDifference(UnitEnd, UnitStart, 8);
  } else {
    Asm.emitLabelDifference(UnitEnd, UnitStart, 4);
  }
  Asm.OutStreamer->emitLabel(UnitStart);
  Asm.emitInt16(Version);
  if (Version == 5) {
    Asm.emitInt8(AddrSize);
    Asm.emitInt8(SegSize);
  }

  MCSymbol *PrologueStart = Asm.createTempSymbol("line_prologue_start");
  MCSymbol *PrologueEnd = Asm.createTempSymbol("line_prologue_end");
  Asm.emitLabelDifference(PrologueEnd, PrologueStart,
                          Format == DwarfFormat::DWARF64 ? 8 : 4);
  Asm.OutStreamer->emitLabel(PrologueStart);

  DWARFDebugLine::Prologue DefaultPrologue = createBasicPrologue();
  writeProloguePayload(DefaultPrologue, Asm);
  Asm.OutStreamer->emitLabel(PrologueEnd);
  return UnitEnd;
}

void dwarfgen::LineTable::writePrologue(AsmPrinter &Asm) const {
  if (Format == DwarfFormat::DWARF64) {
    Asm.emitInt32((int)dwarf::DW_LENGTH_DWARF64);
    Asm.emitInt64(Prologue->TotalLength);
  } else {
    Asm.emitInt32(Prologue->TotalLength);
  }
  Asm.emitInt16(Prologue->getVersion());
  if (Version == 5) {
    Asm.emitInt8(Prologue->getAddressSize());
    Asm.emitInt8(Prologue->SegSelectorSize);
  }
  if (Format == DwarfFormat::DWARF64)
    Asm.emitInt64(Prologue->PrologueLength);
  else
    Asm.emitInt32(Prologue->PrologueLength);

  writeProloguePayload(*Prologue, Asm);
}

static void writeCString(StringRef Str, AsmPrinter &Asm) {
  Asm.OutStreamer->emitBytes(Str);
  Asm.emitInt8(0);
}

static void writeV2IncludeAndFileTable(const DWARFDebugLine::Prologue &Prologue,
                                       AsmPrinter &Asm) {
  for (auto Include : Prologue.IncludeDirectories)
    writeCString(*toString(Include), Asm);

  Asm.emitInt8(0);

  for (auto File : Prologue.FileNames) {
    writeCString(*toString(File.Name), Asm);
    Asm.emitULEB128(File.DirIdx);
    Asm.emitULEB128(File.ModTime);
    Asm.emitULEB128(File.Length);
  }
  Asm.emitInt8(0);
}

static void writeV5IncludeAndFileTable(const DWARFDebugLine::Prologue &Prologue,
                                       AsmPrinter &Asm) {
  Asm.emitInt8(1); // directory_entry_format_count.
  // TODO: Add support for other content descriptions - we currently only
  // support a single DW_LNCT_path/DW_FORM_string.
  Asm.emitULEB128(DW_LNCT_path);
  Asm.emitULEB128(DW_FORM_string);
  Asm.emitULEB128(Prologue.IncludeDirectories.size());
  for (auto Include : Prologue.IncludeDirectories)
    writeCString(*toString(Include), Asm);

  Asm.emitInt8(2); // file_name_entry_format_count.
  Asm.emitULEB128(DW_LNCT_path);
  Asm.emitULEB128(DW_FORM_string);
  Asm.emitULEB128(DW_LNCT_directory_index);
  Asm.emitULEB128(DW_FORM_data1);
  Asm.emitULEB128(Prologue.FileNames.size());
  for (auto File : Prologue.FileNames) {
    writeCString(*toString(File.Name), Asm);
    Asm.emitInt8(File.DirIdx);
  }
}

void dwarfgen::LineTable::writeProloguePayload(
    const DWARFDebugLine::Prologue &Prologue, AsmPrinter &Asm) const {
  Asm.emitInt8(Prologue.MinInstLength);
  if (Version >= 4)
    Asm.emitInt8(Prologue.MaxOpsPerInst);
  Asm.emitInt8(Prologue.DefaultIsStmt);
  Asm.emitInt8(Prologue.LineBase);
  Asm.emitInt8(Prologue.LineRange);
  Asm.emitInt8(Prologue.OpcodeBase);
  for (auto Length : Prologue.StandardOpcodeLengths) {
    Asm.emitInt8(Length);
  }

  if (Version < 5)
    writeV2IncludeAndFileTable(Prologue, Asm);
  else
    writeV5IncludeAndFileTable(Prologue, Asm);
}

//===----------------------------------------------------------------------===//
/// dwarfgen::Generator implementation.
//===----------------------------------------------------------------------===//

dwarfgen::Generator::Generator()
    : MAB(nullptr), MCE(nullptr), MS(nullptr), TLOF(nullptr),
      StringPool(nullptr), Abbreviations(Allocator),
      StringOffsetsStartSym(nullptr), Version(0) {}
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

  MCTargetOptions MCOptions = mc::InitMCTargetOptionsFromFlags();
  MAI.reset(TheTarget->createMCAsmInfo(*MRI, TripleName, MCOptions));
  if (!MAI)
    return make_error<StringError>("no asm info for target " + TripleName,
                                   inconvertibleErrorCode());

  MSTI.reset(TheTarget->createMCSubtargetInfo(TripleName, "", ""));
  if (!MSTI)
    return make_error<StringError>("no subtarget info for target " + TripleName,
                                   inconvertibleErrorCode());

  MAB = TheTarget->createMCAsmBackend(*MSTI, *MRI, MCOptions);
  if (!MAB)
    return make_error<StringError>("no asm backend for target " + TripleName,
                                   inconvertibleErrorCode());

  MII.reset(TheTarget->createMCInstrInfo());
  if (!MII)
    return make_error<StringError>("no instr info info for target " +
                                       TripleName,
                                   inconvertibleErrorCode());

  TM.reset(TheTarget->createTargetMachine(TripleName, "", "", TargetOptions(),
                                          std::nullopt));
  if (!TM)
    return make_error<StringError>("no target machine for target " + TripleName,
                                   inconvertibleErrorCode());

  MC.reset(new MCContext(TheTriple, MAI.get(), MRI.get(), MSTI.get()));
  TLOF = TM->getObjFileLowering();
  TLOF->Initialize(*MC, *TM);
  MC->setObjectFileInfo(TLOF);

  MCE = TheTarget->createMCCodeEmitter(*MII, *MC);
  if (!MCE)
    return make_error<StringError>("no code emitter for target " + TripleName,
                                   inconvertibleErrorCode());

  Stream = std::make_unique<raw_svector_ostream>(FileBytes);

  MS = TheTarget->createMCObjectStreamer(
      TheTriple, *MC, std::unique_ptr<MCAsmBackend>(MAB),
      MAB->createObjectWriter(*Stream), std::unique_ptr<MCCodeEmitter>(MCE),
      *MSTI, MCOptions.MCRelaxAll, MCOptions.MCIncrementalLinkerCompatible,
      /*DWARFMustBeAtTheEnd*/ false);
  if (!MS)
    return make_error<StringError>("no object streamer for target " +
                                       TripleName,
                                   inconvertibleErrorCode());


  // Finally create the AsmPrinter we'll use to emit the DIEs.
  Asm.reset(TheTarget->createAsmPrinter(*TM, std::unique_ptr<MCStreamer>(MS)));
  if (!Asm)
    return make_error<StringError>("no asm printer for target " + TripleName,
                                   inconvertibleErrorCode());

  // Set the DWARF version correctly on all classes that we use.
  MC->setDwarfVersion(Version);
  Asm->setDwarfVersion(Version);

  StringPool = std::make_unique<DwarfStringPool>(Allocator, *Asm, StringRef());
  StringOffsetsStartSym = Asm->createTempSymbol("str_offsets_base");

  AddrTableStartSym = Asm->createTempSymbol("addr_table_base");

  return Error::success();
}

unsigned dwarfgen::Generator::DummyAddressPool::getIndex(uint64_t Address) {
  AddressValues.push_back(Address);
  return static_cast<unsigned>(AddressValues.size() - 1);
}

void dwarfgen::Generator::DummyAddressPool::emit(AsmPrinter &Asm,
                                                 MCSection *AddrSection,
                                                 MCSymbol *StartSym) {
  const uint8_t AddrSize = Asm.getPointerSize();

  // Switch to .debug_addr section
  Asm.OutStreamer->switchSection(AddrSection);

  if (Asm.getDwarfVersion() >= 5) {
    // Emit header
    Asm.emitDwarfUnitLength(AddrSize * AddressValues.size() + 4,
                            "Length of contribution");
    Asm.emitInt16(Asm.getDwarfVersion());
    Asm.emitInt8(AddrSize);
    Asm.emitInt8(0);
  }

  if (StartSym)
    Asm.OutStreamer->emitLabel(StartSym);

  // Emit addresses
  for (uint64_t Addr : AddressValues)
    Asm.OutStreamer->emitIntValue(Addr, AddrSize);
}

StringRef dwarfgen::Generator::generate() {
  // Offset from the first CU in the debug info section is 0 initially.
  uint64_t SecOffset = 0;

  // Iterate over each compile unit and set the size and offsets for each
  // DIE within each compile unit. All offsets are CU relative.
  for (auto &CU : CompileUnits) {
    // Set the absolute .debug_info offset for this compile unit.
    CU->setOffset(SecOffset);
    // The DIEs contain compile unit relative offsets and the offset depends
    // on the Dwarf version.
    unsigned CUOffset = 4 + // Length
                        2 + // Version
                        4 + // Abbreviation offset
                        1;  // Address size
    if (Asm->getDwarfVersion() >= 5)
      CUOffset += 1; // DW_UT_compile tag.

    CUOffset = CU->getUnitDIE().computeSizeAndOffsets(CUOffset);
    // Update our absolute .debug_info offset.
    SecOffset += CUOffset;
    unsigned CUOffsetUnitLength = 4;
    CU->setLength(CUOffset - CUOffsetUnitLength);
  }
  Abbreviations.Emit(Asm.get(), TLOF->getDwarfAbbrevSection());

  StringPool->emitStringOffsetsTableHeader(*Asm, TLOF->getDwarfStrOffSection(),
                                           StringOffsetsStartSym);
  StringPool->emit(*Asm, TLOF->getDwarfStrSection(),
                   TLOF->getDwarfStrOffSection());

  if (Asm->getDwarfVersion() >= 5)
    AddressPool.emit(*Asm, TLOF->getDwarfAddrSection(), AddrTableStartSym);

  MS->switchSection(TLOF->getDwarfInfoSection());
  for (auto &CU : CompileUnits) {
    uint16_t Version = CU->getVersion();
    auto Length = CU->getLength();
    MC->setDwarfVersion(Version);
    assert(Length != -1U);
    Asm->emitInt32(Length);
    Asm->emitInt16(Version);
    if (Version <= 4) {
      Asm->emitInt32(0);
      Asm->emitInt8(CU->getAddressSize());
    } else {
      Asm->emitInt8(dwarf::DW_UT_compile);
      Asm->emitInt8(CU->getAddressSize());
      Asm->emitInt32(0);
    }
    Asm->emitDwarfDIE(*CU->getUnitDIE().Die);
  }

  MS->switchSection(TLOF->getDwarfLineSection());
  for (auto &LT : LineTables)
    LT->generate(*MC, *Asm);

  MS->finish();
  if (FileBytes.empty())
    return StringRef();
  return StringRef(FileBytes.data(), FileBytes.size());
}

bool dwarfgen::Generator::saveFile(StringRef Path) {
  if (FileBytes.empty())
    return false;
  std::error_code EC;
  raw_fd_ostream Strm(Path, EC, sys::fs::OF_None);
  if (EC)
    return false;
  Strm.write(FileBytes.data(), FileBytes.size());
  Strm.close();
  return true;
}

dwarfgen::CompileUnit &dwarfgen::Generator::addCompileUnit() {
  CompileUnits.push_back(
      std::make_unique<CompileUnit>(*this, Version, Asm->getPointerSize()));
  return *CompileUnits.back();
}

dwarfgen::LineTable &dwarfgen::Generator::addLineTable(DwarfFormat Format) {
  LineTables.push_back(
      std::make_unique<LineTable>(Version, Format, Asm->getPointerSize()));
  return *LineTables.back();
}
