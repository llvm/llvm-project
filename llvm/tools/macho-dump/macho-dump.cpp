//===-- macho-dump.cpp - Mach Object Dumping Tool -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a testing tool for use with the MC/Mach-O LLVM components.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/MachOObject.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
using namespace llvm;
using namespace llvm::object;

static cl::opt<std::string>
InputFile(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<bool>
ShowSectionData("dump-section-data", cl::desc("Dump the contents of sections"),
                cl::init(false));

///

static const char *ProgramName;

static void Message(const char *Type, const Twine &Msg) {
  errs() << ProgramName << ": " << Type << ": " << Msg << "\n";
}

static int Error(const Twine &Msg) {
  Message("error", Msg);
  return 1;
}

static void Warning(const Twine &Msg) {
  Message("warning", Msg);
}

///

static void DumpSegmentCommandData(StringRef Name,
                                   uint64_t VMAddr, uint64_t VMSize,
                                   uint64_t FileOffset, uint64_t FileSize,
                                   uint32_t MaxProt, uint32_t InitProt,
                                   uint32_t NumSections, uint32_t Flags) {
  outs() << "  ('segment_name', '";
  outs().write_escaped(Name, /*UseHexEscapes=*/true) << "')\n";
  outs() << "  ('vm_addr', " << VMAddr << ")\n";
  outs() << "  ('vm_size', " << VMSize << ")\n";
  outs() << "  ('file_offset', " << FileOffset << ")\n";
  outs() << "  ('file_size', " << FileSize << ")\n";
  outs() << "  ('maxprot', " << MaxProt << ")\n";
  outs() << "  ('initprot', " << InitProt << ")\n";
  outs() << "  ('num_sections', " << NumSections << ")\n";
  outs() << "  ('flags', " << Flags << ")\n";
}

static int DumpSectionData(MachOObject &Obj, unsigned Index, StringRef Name,
                           StringRef SegmentName, uint64_t Address,
                           uint64_t Size, uint32_t Offset,
                           uint32_t Align, uint32_t RelocationTableOffset,
                           uint32_t NumRelocationTableEntries,
                           uint32_t Flags, uint32_t Reserved1,
                           uint32_t Reserved2, uint64_t Reserved3 = ~0ULL) {
  outs() << "    # Section " << Index << "\n";
  outs() << "   (('section_name', '";
  outs().write_escaped(Name, /*UseHexEscapes=*/true) << "')\n";
  outs() << "    ('segment_name', '";
  outs().write_escaped(SegmentName, /*UseHexEscapes=*/true) << "')\n";
  outs() << "    ('address', " << Address << ")\n";
  outs() << "    ('size', " << Size << ")\n";
  outs() << "    ('offset', " << Offset << ")\n";
  outs() << "    ('alignment', " << Align << ")\n";
  outs() << "    ('reloc_offset', " << RelocationTableOffset << ")\n";
  outs() << "    ('num_reloc', " << NumRelocationTableEntries << ")\n";
  outs() << "    ('flags', " << format("0x%x", Flags) << ")\n";
  outs() << "    ('reserved1', " << Reserved1 << ")\n";
  outs() << "    ('reserved2', " << Reserved2 << ")\n";
  if (Reserved3 != ~0ULL)
    outs() << "    ('reserved3', " << Reserved3 << ")\n";
  outs() << "   ),\n";

  // Dump the relocation entries.
  int Res = 0;
  outs() << "  ('_relocations', [\n";
  for (unsigned i = 0; i != NumRelocationTableEntries; ++i) {
    InMemoryStruct<macho::RelocationEntry> RE;
    Obj.ReadRelocationEntry(RelocationTableOffset, i, RE);
    if (!RE) {
      Res = Error("unable to read relocation table entry '" + Twine(i) + "'");
      break;
    }
    
    outs() << "    # Relocation " << i << "\n";
    outs() << "    (('word-0', " << format("0x%x", RE->Word0) << "),\n";
    outs() << "     ('word-1', " << format("0x%x", RE->Word1) << ")),\n";
  }
  outs() << "  ])\n";

  // Dump the section data, if requested.
  if (ShowSectionData) {
    outs() << "  ('_section_data', '";
    StringRef Data = Obj.getData(Offset, Size);
    for (unsigned i = 0; i != Data.size(); ++i) {
      if (i && (i % 4) == 0)
        outs() << ' ';
      outs() << hexdigit((Data[i] >> 4) & 0xF, /*LowerCase=*/true);
      outs() << hexdigit((Data[i] >> 0) & 0xF, /*LowerCase=*/true);
    }
    outs() << "')\n";
  }

  return Res;
}

static int DumpSegmentCommand(MachOObject &Obj,
                               const MachOObject::LoadCommandInfo &LCI) {
  InMemoryStruct<macho::SegmentLoadCommand> SLC;
  Obj.ReadSegmentLoadCommand(LCI, SLC);
  if (!SLC)
    return Error("unable to read segment load command");

  DumpSegmentCommandData(StringRef(SLC->Name, 16), SLC->VMAddress,
                         SLC->VMSize, SLC->FileOffset, SLC->FileSize,
                         SLC->MaxVMProtection, SLC->InitialVMProtection,
                         SLC->NumSections, SLC->Flags);

  // Dump the sections.
  int Res = 0;
  outs() << "  ('sections', [\n";
  for (unsigned i = 0; i != SLC->NumSections; ++i) {
    InMemoryStruct<macho::Section> Sect;
    Obj.ReadSection(LCI, i, Sect);
    if (!SLC) {
      Res = Error("unable to read section '" + Twine(i) + "'");
      break;
    }

    if ((Res = DumpSectionData(Obj, i, StringRef(Sect->Name, 16),
                               StringRef(Sect->SegmentName, 16), Sect->Address,
                               Sect->Size, Sect->Offset, Sect->Align,
                               Sect->RelocationTableOffset,
                               Sect->NumRelocationTableEntries, Sect->Flags,
                               Sect->Reserved1, Sect->Reserved2)))
      break;
  }
  outs() << "  ])\n";

  return Res;
}

static int DumpSegment64Command(MachOObject &Obj,
                               const MachOObject::LoadCommandInfo &LCI) {
  InMemoryStruct<macho::Segment64LoadCommand> SLC;
  Obj.ReadSegment64LoadCommand(LCI, SLC);
  if (!SLC)
    return Error("unable to read segment load command");

  DumpSegmentCommandData(StringRef(SLC->Name, 16), SLC->VMAddress,
                         SLC->VMSize, SLC->FileOffset, SLC->FileSize,
                         SLC->MaxVMProtection, SLC->InitialVMProtection,
                         SLC->NumSections, SLC->Flags);

  // Dump the sections.
  int Res = 0;
  outs() << "  ('sections', [\n";
  for (unsigned i = 0; i != SLC->NumSections; ++i) {
    InMemoryStruct<macho::Section64> Sect;
    Obj.ReadSection64(LCI, i, Sect);
    if (!SLC) {
      Res = Error("unable to read section '" + Twine(i) + "'");
      break;
    }

    if ((Res = DumpSectionData(Obj, i, StringRef(Sect->Name, 16),
                               StringRef(Sect->SegmentName, 16), Sect->Address,
                               Sect->Size, Sect->Offset, Sect->Align,
                               Sect->RelocationTableOffset,
                               Sect->NumRelocationTableEntries, Sect->Flags,
                               Sect->Reserved1, Sect->Reserved2,
                               Sect->Reserved3)))
      break;
  }
  outs() << "  ])\n";

  return Res;
}

static void DumpSymbolTableEntryData(MachOObject &Obj,
                                     unsigned Index, uint32_t StringIndex,
                                     uint8_t Type, uint8_t SectionIndex,
                                     uint16_t Flags, uint64_t Value) {
  outs() << "    # Symbol " << Index << "\n";
  outs() << "   (('n_strx', " << StringIndex << ")\n";
  outs() << "    ('n_type', " << format("0x%x", Type) << ")\n";
  outs() << "    ('n_sect', " << uint32_t(SectionIndex) << ")\n";
  outs() << "    ('n_desc', " << Flags << ")\n";
  outs() << "    ('n_value', " << Value << ")\n";
  outs() << "    ('_string', '" << Obj.getStringAtIndex(StringIndex) << "')\n";
  outs() << "   ),\n";
}

static int DumpSymtabCommand(MachOObject &Obj,
                             const MachOObject::LoadCommandInfo &LCI) {
  InMemoryStruct<macho::SymtabLoadCommand> SLC;
  Obj.ReadSymtabLoadCommand(LCI, SLC);
  if (!SLC)
    return Error("unable to read segment load command");

  outs() << "  ('symoff', " << SLC->SymbolTableOffset << ")\n";
  outs() << "  ('nsyms', " << SLC->NumSymbolTableEntries << ")\n";
  outs() << "  ('stroff', " << SLC->StringTableOffset << ")\n";
  outs() << "  ('strsize', " << SLC->StringTableSize << ")\n";

  // Cache the string table data.
  Obj.RegisterStringTable(*SLC);

  // Dump the string data.
  outs() << "  ('_string_data', '";
  outs().write_escaped(Obj.getStringTableData(),
                       /*UseHexEscapes=*/true) << "')\n";

  // Dump the symbol table.
  int Res = 0;
  outs() << "  ('_symbols', [\n";
  for (unsigned i = 0; i != SLC->NumSymbolTableEntries; ++i) {
    if (Obj.is64Bit()) {
      InMemoryStruct<macho::Symbol64TableEntry> STE;
      Obj.ReadSymbol64TableEntry(SLC->SymbolTableOffset, i, STE);
      if (!STE) {
        Res = Error("unable to read symbol: '" + Twine(i) + "'");
        break;
      }

      DumpSymbolTableEntryData(Obj, i, STE->StringIndex, STE->Type,
                               STE->SectionIndex, STE->Flags, STE->Value);
    } else {
      InMemoryStruct<macho::SymbolTableEntry> STE;
      Obj.ReadSymbolTableEntry(SLC->SymbolTableOffset, i, STE);
      if (!SLC) {
        Res = Error("unable to read symbol: '" + Twine(i) + "'");
        break;
      }

      DumpSymbolTableEntryData(Obj, i, STE->StringIndex, STE->Type,
                               STE->SectionIndex, STE->Flags, STE->Value);
    }
  }
  outs() << "  ])\n";

  return Res;
}

static int DumpDysymtabCommand(MachOObject &Obj,
                             const MachOObject::LoadCommandInfo &LCI) {
  InMemoryStruct<macho::DysymtabLoadCommand> DLC;
  Obj.ReadDysymtabLoadCommand(LCI, DLC);
  if (!DLC)
    return Error("unable to read segment load command");

  outs() << "  ('ilocalsym', " << DLC->LocalSymbolsIndex << ")\n";
  outs() << "  ('nlocalsym', " << DLC->NumLocalSymbols << ")\n";
  outs() << "  ('iextdefsym', " << DLC->ExternalSymbolsIndex << ")\n";
  outs() << "  ('nextdefsym', " << DLC->NumExternalSymbols << ")\n";
  outs() << "  ('iundefsym', " << DLC->UndefinedSymbolsIndex << ")\n";
  outs() << "  ('nundefsym', " << DLC->NumUndefinedSymbols << ")\n";
  outs() << "  ('tocoff', " << DLC->TOCOffset << ")\n";
  outs() << "  ('ntoc', " << DLC->NumTOCEntries << ")\n";
  outs() << "  ('modtaboff', " << DLC->ModuleTableOffset << ")\n";
  outs() << "  ('nmodtab', " << DLC->NumModuleTableEntries << ")\n";
  outs() << "  ('extrefsymoff', " << DLC->ReferenceSymbolTableOffset << ")\n";
  outs() << "  ('nextrefsyms', "
         << DLC->NumReferencedSymbolTableEntries << ")\n";
  outs() << "  ('indirectsymoff', " << DLC->IndirectSymbolTableOffset << ")\n";
  outs() << "  ('nindirectsyms', "
         << DLC->NumIndirectSymbolTableEntries << ")\n";
  outs() << "  ('extreloff', " << DLC->ExternalRelocationTableOffset << ")\n";
  outs() << "  ('nextrel', " << DLC->NumExternalRelocationTableEntries << ")\n";
  outs() << "  ('locreloff', " << DLC->LocalRelocationTableOffset << ")\n";
  outs() << "  ('nlocrel', " << DLC->NumLocalRelocationTableEntries << ")\n";

  // Dump the indirect symbol table.
  int Res = 0;
  outs() << "  ('_indirect_symbols', [\n";
  for (unsigned i = 0; i != DLC->NumIndirectSymbolTableEntries; ++i) {
    InMemoryStruct<macho::IndirectSymbolTableEntry> ISTE;
    Obj.ReadIndirectSymbolTableEntry(*DLC, i, ISTE);
    if (!ISTE) {
      Res = Error("unable to read segment load command");
      break;
    }

    outs() << "    # Indirect Symbol " << i << "\n";
    outs() << "    (('symbol_index', "
           << format("0x%x", ISTE->Index) << "),),\n";
  }
  outs() << "  ])\n";

  return Res;
}

static int DumpLinkeditDataCommand(MachOObject &Obj,
                                   const MachOObject::LoadCommandInfo &LCI) {
  InMemoryStruct<macho::LinkeditDataLoadCommand> LLC;
  Obj.ReadLinkeditDataLoadCommand(LCI, LLC);
  if (!LLC)
    return Error("unable to read segment load command");

  outs() << "  ('dataoff', " << LLC->DataOffset << ")\n"
         << "  ('datasize', " << LLC->DataSize << ")\n"
         << "  ('_addresses', [\n";

  SmallVector<uint64_t, 8> Addresses;
  Obj.ReadULEB128s(LLC->DataOffset, Addresses);
  for (unsigned i = 0, e = Addresses.size(); i != e; ++i)
    outs() << "    # Address " << i << '\n'
           << "    ('address', " << format("0x%x", Addresses[i]) << "),\n";

  outs() << "  ])\n";

  return 0;
}

static int DumpDataInCodeDataCommand(MachOObject &Obj,
                                     const MachOObject::LoadCommandInfo &LCI) {
  InMemoryStruct<macho::LinkeditDataLoadCommand> LLC;
  Obj.ReadLinkeditDataLoadCommand(LCI, LLC);
  if (!LLC)
    return Error("unable to read segment load command");

  outs() << "  ('dataoff', " << LLC->DataOffset << ")\n"
         << "  ('datasize', " << LLC->DataSize << ")\n"
         << "  ('_data_regions', [\n";


  unsigned NumRegions = LLC->DataSize / 8;
  for (unsigned i = 0; i < NumRegions; ++i) {
    InMemoryStruct<macho::DataInCodeTableEntry> DICE;
    Obj.ReadDataInCodeTableEntry(LLC->DataOffset, i, DICE);
    if (!DICE)
      return Error("unable to read DataInCodeTableEntry");
    outs() << "    # DICE " << i << "\n"
           << "    ('offset', " << DICE->Offset << ")\n"
           << "    ('length', " << DICE->Length << ")\n"
           << "    ('kind', " << DICE->Kind << ")\n";
  }

  outs() <<"  ])\n";

  return 0;
}


static int DumpLoadCommand(MachOObject &Obj, unsigned Index) {
  const MachOObject::LoadCommandInfo &LCI = Obj.getLoadCommandInfo(Index);
  int Res = 0;

  outs() << "  # Load Command " << Index << "\n"
         << " (('command', " << LCI.Command.Type << ")\n"
         << "  ('size', " << LCI.Command.Size << ")\n";
  switch (LCI.Command.Type) {
  case macho::LCT_Segment:
    Res = DumpSegmentCommand(Obj, LCI);
    break;
  case macho::LCT_Segment64:
    Res = DumpSegment64Command(Obj, LCI);
    break;
  case macho::LCT_Symtab:
    Res = DumpSymtabCommand(Obj, LCI);
    break;
  case macho::LCT_Dysymtab:
    Res = DumpDysymtabCommand(Obj, LCI);
    break;
  case macho::LCT_CodeSignature:
  case macho::LCT_SegmentSplitInfo:
  case macho::LCT_FunctionStarts:
    Res = DumpLinkeditDataCommand(Obj, LCI);
    break;
  case macho::LCT_DataInCode:
    Res = DumpDataInCodeDataCommand(Obj, LCI);
    break;
  default:
    Warning("unknown load command: " + Twine(LCI.Command.Type));
    break;
  }
  outs() << " ),\n";

  return Res;
}

int main(int argc, char **argv) {
  ProgramName = argv[0];
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "llvm Mach-O dumping tool\n");

  // Load the input file.
  std::string ErrorStr;
  OwningPtr<MemoryBuffer> InputBuffer;
  if (error_code ec = MemoryBuffer::getFileOrSTDIN(InputFile, InputBuffer))
    return Error("unable to read input: '" + ec.message() + "'");

  // Construct the Mach-O wrapper object.
  OwningPtr<MachOObject> InputObject(
    MachOObject::LoadFromBuffer(InputBuffer.take(), &ErrorStr));
  if (!InputObject)
    return Error("unable to load object: '" + ErrorStr + "'");

  // Print the header
  InputObject->printHeader(outs());

  // Print the load commands.
  int Res = 0;
  outs() << "('load_commands', [\n";
  for (unsigned i = 0; i != InputObject->getHeader().NumLoadCommands; ++i)
    if ((Res = DumpLoadCommand(*InputObject, i)))
      break;
  outs() << "])\n";

  return Res;
}
