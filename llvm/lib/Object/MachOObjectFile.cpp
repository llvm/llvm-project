//===- MachOObjectFile.cpp - Mach-O object file binding ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MachOObjectFile class, which binds the MachOObject
// class to the generic ObjectFile wrapper.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Triple.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/MachOFormat.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cctype>
#include <cstring>
#include <limits>

using namespace llvm;
using namespace object;

namespace llvm {
namespace object {

MachOObjectFile::MachOObjectFile(MemoryBuffer *Object, MachOObject *MOO,
                                 error_code &ec)
    : ObjectFile(Binary::ID_MachO, Object, ec),
      MachOObj(MOO),
      RegisteredStringTable(std::numeric_limits<uint32_t>::max()) {
  DataRefImpl DRI;
  moveToNextSection(DRI);
  uint32_t LoadCommandCount = MachOObj->getHeader().NumLoadCommands;
  while (DRI.d.a < LoadCommandCount) {
    Sections.push_back(DRI);
    DRI.d.b++;
    moveToNextSection(DRI);
  }
}


ObjectFile *ObjectFile::createMachOObjectFile(MemoryBuffer *Buffer) {
  error_code ec;
  std::string Err;
  MachOObject *MachOObj = MachOObject::LoadFromBuffer(Buffer, &Err);
  if (!MachOObj)
    return NULL;
  return new MachOObjectFile(Buffer, MachOObj, ec);
}

/*===-- Symbols -----------------------------------------------------------===*/

void MachOObjectFile::moveToNextSymbol(DataRefImpl &DRI) const {
  uint32_t LoadCommandCount = MachOObj->getHeader().NumLoadCommands;
  while (DRI.d.a < LoadCommandCount) {
    LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
    if (LCI.Command.Type == macho::LCT_Symtab) {
      InMemoryStruct<macho::SymtabLoadCommand> SymtabLoadCmd;
      MachOObj->ReadSymtabLoadCommand(LCI, SymtabLoadCmd);
      if (DRI.d.b < SymtabLoadCmd->NumSymbolTableEntries)
        return;
    }

    DRI.d.a++;
    DRI.d.b = 0;
  }
}

void MachOObjectFile::getSymbolTableEntry(DataRefImpl DRI,
    InMemoryStruct<macho::SymbolTableEntry> &Res) const {
  InMemoryStruct<macho::SymtabLoadCommand> SymtabLoadCmd;
  LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
  MachOObj->ReadSymtabLoadCommand(LCI, SymtabLoadCmd);

  if (RegisteredStringTable != DRI.d.a) {
    MachOObj->RegisterStringTable(*SymtabLoadCmd);
    RegisteredStringTable = DRI.d.a;
  }

  MachOObj->ReadSymbolTableEntry(SymtabLoadCmd->SymbolTableOffset, DRI.d.b,
                                 Res);
}

void MachOObjectFile::getSymbol64TableEntry(DataRefImpl DRI,
    InMemoryStruct<macho::Symbol64TableEntry> &Res) const {
  InMemoryStruct<macho::SymtabLoadCommand> SymtabLoadCmd;
  LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
  MachOObj->ReadSymtabLoadCommand(LCI, SymtabLoadCmd);

  if (RegisteredStringTable != DRI.d.a) {
    MachOObj->RegisterStringTable(*SymtabLoadCmd);
    RegisteredStringTable = DRI.d.a;
  }

  MachOObj->ReadSymbol64TableEntry(SymtabLoadCmd->SymbolTableOffset, DRI.d.b,
                                   Res);
}


error_code MachOObjectFile::getSymbolNext(DataRefImpl DRI,
                                          SymbolRef &Result) const {
  DRI.d.b++;
  moveToNextSymbol(DRI);
  Result = SymbolRef(DRI, this);
  return object_error::success;
}

error_code MachOObjectFile::getSymbolName(DataRefImpl DRI,
                                          StringRef &Result) const {
  if (MachOObj->is64Bit()) {
    InMemoryStruct<macho::Symbol64TableEntry> Entry;
    getSymbol64TableEntry(DRI, Entry);
    Result = MachOObj->getStringAtIndex(Entry->StringIndex);
  } else {
    InMemoryStruct<macho::SymbolTableEntry> Entry;
    getSymbolTableEntry(DRI, Entry);
    Result = MachOObj->getStringAtIndex(Entry->StringIndex);
  }
  return object_error::success;
}

error_code MachOObjectFile::getSymbolFileOffset(DataRefImpl DRI,
                                                uint64_t &Result) const {
  if (MachOObj->is64Bit()) {
    InMemoryStruct<macho::Symbol64TableEntry> Entry;
    getSymbol64TableEntry(DRI, Entry);
    Result = Entry->Value;
    if (Entry->SectionIndex) {
      InMemoryStruct<macho::Section64> Section;
      getSection64(Sections[Entry->SectionIndex-1], Section);
      Result += Section->Offset - Section->Address;
    }
  } else {
    InMemoryStruct<macho::SymbolTableEntry> Entry;
    getSymbolTableEntry(DRI, Entry);
    Result = Entry->Value;
    if (Entry->SectionIndex) {
      InMemoryStruct<macho::Section> Section;
      getSection(Sections[Entry->SectionIndex-1], Section);
      Result += Section->Offset - Section->Address;
    }
  }

  return object_error::success;
}

error_code MachOObjectFile::getSymbolAddress(DataRefImpl DRI,
                                             uint64_t &Result) const {
  if (MachOObj->is64Bit()) {
    InMemoryStruct<macho::Symbol64TableEntry> Entry;
    getSymbol64TableEntry(DRI, Entry);
    Result = Entry->Value;
  } else {
    InMemoryStruct<macho::SymbolTableEntry> Entry;
    getSymbolTableEntry(DRI, Entry);
    Result = Entry->Value;
  }
  return object_error::success;
}

error_code MachOObjectFile::getSymbolSize(DataRefImpl DRI,
                                          uint64_t &Result) const {
  uint32_t LoadCommandCount = MachOObj->getHeader().NumLoadCommands;
  uint64_t BeginOffset;
  uint64_t EndOffset = 0;
  uint8_t SectionIndex;
  if (MachOObj->is64Bit()) {
    InMemoryStruct<macho::Symbol64TableEntry> Entry;
    getSymbol64TableEntry(DRI, Entry);
    BeginOffset = Entry->Value;
    SectionIndex = Entry->SectionIndex;
    if (!SectionIndex) {
      uint32_t flags = SymbolRef::SF_None;
      getSymbolFlags(DRI, flags);
      if (flags & SymbolRef::SF_Common)
        Result = Entry->Value;
      else
        Result = UnknownAddressOrSize;
      return object_error::success;
    }
    // Unfortunately symbols are unsorted so we need to touch all
    // symbols from load command
    DRI.d.b = 0;
    uint32_t Command = DRI.d.a;
    while (Command == DRI.d.a) {
      moveToNextSymbol(DRI);
      if (DRI.d.a < LoadCommandCount) {
        getSymbol64TableEntry(DRI, Entry);
        if (Entry->SectionIndex == SectionIndex && Entry->Value > BeginOffset)
          if (!EndOffset || Entry->Value < EndOffset)
            EndOffset = Entry->Value;
      }
      DRI.d.b++;
    }
  } else {
    InMemoryStruct<macho::SymbolTableEntry> Entry;
    getSymbolTableEntry(DRI, Entry);
    BeginOffset = Entry->Value;
    SectionIndex = Entry->SectionIndex;
    if (!SectionIndex) {
      uint32_t flags = SymbolRef::SF_None;
      getSymbolFlags(DRI, flags);
      if (flags & SymbolRef::SF_Common)
        Result = Entry->Value;
      else
        Result = UnknownAddressOrSize;
      return object_error::success;
    }
    // Unfortunately symbols are unsorted so we need to touch all
    // symbols from load command
    DRI.d.b = 0;
    uint32_t Command = DRI.d.a;
    while (Command == DRI.d.a) {
      moveToNextSymbol(DRI);
      if (DRI.d.a < LoadCommandCount) {
        getSymbolTableEntry(DRI, Entry);
        if (Entry->SectionIndex == SectionIndex && Entry->Value > BeginOffset)
          if (!EndOffset || Entry->Value < EndOffset)
            EndOffset = Entry->Value;
      }
      DRI.d.b++;
    }
  }
  if (!EndOffset) {
    uint64_t Size;
    getSectionSize(Sections[SectionIndex-1], Size);
    getSectionAddress(Sections[SectionIndex-1], EndOffset);
    EndOffset += Size;
  }
  Result = EndOffset - BeginOffset;
  return object_error::success;
}

error_code MachOObjectFile::getSymbolNMTypeChar(DataRefImpl DRI,
                                                char &Result) const {
  uint8_t Type, Flags;
  if (MachOObj->is64Bit()) {
    InMemoryStruct<macho::Symbol64TableEntry> Entry;
    getSymbol64TableEntry(DRI, Entry);
    Type = Entry->Type;
    Flags = Entry->Flags;
  } else {
    InMemoryStruct<macho::SymbolTableEntry> Entry;
    getSymbolTableEntry(DRI, Entry);
    Type = Entry->Type;
    Flags = Entry->Flags;
  }

  char Char;
  switch (Type & macho::STF_TypeMask) {
    case macho::STT_Undefined:
      Char = 'u';
      break;
    case macho::STT_Absolute:
    case macho::STT_Section:
      Char = 's';
      break;
    default:
      Char = '?';
      break;
  }

  if (Flags & (macho::STF_External | macho::STF_PrivateExtern))
    Char = toupper(Char);
  Result = Char;
  return object_error::success;
}

error_code MachOObjectFile::getSymbolFlags(DataRefImpl DRI,
                                           uint32_t &Result) const {
  uint16_t MachOFlags;
  uint8_t MachOType;
  if (MachOObj->is64Bit()) {
    InMemoryStruct<macho::Symbol64TableEntry> Entry;
    getSymbol64TableEntry(DRI, Entry);
    MachOFlags = Entry->Flags;
    MachOType = Entry->Type;
  } else {
    InMemoryStruct<macho::SymbolTableEntry> Entry;
    getSymbolTableEntry(DRI, Entry);
    MachOFlags = Entry->Flags;
    MachOType = Entry->Type;
  }

  // TODO: Correctly set SF_ThreadLocal
  Result = SymbolRef::SF_None;

  if ((MachOType & MachO::NlistMaskType) == MachO::NListTypeUndefined)
    Result |= SymbolRef::SF_Undefined;

  if (MachOFlags & macho::STF_StabsEntryMask)
    Result |= SymbolRef::SF_FormatSpecific;

  if (MachOType & MachO::NlistMaskExternal) {
    Result |= SymbolRef::SF_Global;
    if ((MachOType & MachO::NlistMaskType) == MachO::NListTypeUndefined)
      Result |= SymbolRef::SF_Common;
  }

  if (MachOFlags & (MachO::NListDescWeakRef | MachO::NListDescWeakDef))
    Result |= SymbolRef::SF_Weak;

  if ((MachOType & MachO::NlistMaskType) == MachO::NListTypeAbsolute)
    Result |= SymbolRef::SF_Absolute;

  return object_error::success;
}

error_code MachOObjectFile::getSymbolSection(DataRefImpl Symb,
                                             section_iterator &Res) const {
  uint8_t index;
  if (MachOObj->is64Bit()) {
    InMemoryStruct<macho::Symbol64TableEntry> Entry;
    getSymbol64TableEntry(Symb, Entry);
    index = Entry->SectionIndex;
  } else {
    InMemoryStruct<macho::SymbolTableEntry> Entry;
    getSymbolTableEntry(Symb, Entry);
    index = Entry->SectionIndex;
  }

  if (index == 0)
    Res = end_sections();
  else
    Res = section_iterator(SectionRef(Sections[index-1], this));

  return object_error::success;
}

error_code MachOObjectFile::getSymbolType(DataRefImpl Symb,
                                          SymbolRef::Type &Res) const {
  uint8_t n_type;
  if (MachOObj->is64Bit()) {
    InMemoryStruct<macho::Symbol64TableEntry> Entry;
    getSymbol64TableEntry(Symb, Entry);
    n_type = Entry->Type;
  } else {
    InMemoryStruct<macho::SymbolTableEntry> Entry;
    getSymbolTableEntry(Symb, Entry);
    n_type = Entry->Type;
  }
  Res = SymbolRef::ST_Other;

  // If this is a STAB debugging symbol, we can do nothing more.
  if (n_type & MachO::NlistMaskStab) {
    Res = SymbolRef::ST_Debug;
    return object_error::success;
  }

  switch (n_type & MachO::NlistMaskType) {
    case MachO::NListTypeUndefined :
      Res = SymbolRef::ST_Unknown;
      break;
    case MachO::NListTypeSection :
      Res = SymbolRef::ST_Function;
      break;
  }
  return object_error::success;
}

error_code MachOObjectFile::getSymbolValue(DataRefImpl Symb,
                                           uint64_t &Val) const {
  report_fatal_error("getSymbolValue unimplemented in MachOObjectFile");
}

symbol_iterator MachOObjectFile::begin_symbols() const {
  // DRI.d.a = segment number; DRI.d.b = symbol index.
  DataRefImpl DRI;
  moveToNextSymbol(DRI);
  return symbol_iterator(SymbolRef(DRI, this));
}

symbol_iterator MachOObjectFile::end_symbols() const {
  DataRefImpl DRI;
  DRI.d.a = MachOObj->getHeader().NumLoadCommands;
  return symbol_iterator(SymbolRef(DRI, this));
}

symbol_iterator MachOObjectFile::begin_dynamic_symbols() const {
  // TODO: implement
  report_fatal_error("Dynamic symbols unimplemented in MachOObjectFile");
}

symbol_iterator MachOObjectFile::end_dynamic_symbols() const {
  // TODO: implement
  report_fatal_error("Dynamic symbols unimplemented in MachOObjectFile");
}

library_iterator MachOObjectFile::begin_libraries_needed() const {
  // TODO: implement
  report_fatal_error("Needed libraries unimplemented in MachOObjectFile");
}

library_iterator MachOObjectFile::end_libraries_needed() const {
  // TODO: implement
  report_fatal_error("Needed libraries unimplemented in MachOObjectFile");
}

StringRef MachOObjectFile::getLoadName() const {
  // TODO: Implement
  report_fatal_error("get_load_name() unimplemented in MachOObjectFile");
}

/*===-- Sections ----------------------------------------------------------===*/

void MachOObjectFile::moveToNextSection(DataRefImpl &DRI) const {
  uint32_t LoadCommandCount = MachOObj->getHeader().NumLoadCommands;
  while (DRI.d.a < LoadCommandCount) {
    LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
    if (LCI.Command.Type == macho::LCT_Segment) {
      InMemoryStruct<macho::SegmentLoadCommand> SegmentLoadCmd;
      MachOObj->ReadSegmentLoadCommand(LCI, SegmentLoadCmd);
      if (DRI.d.b < SegmentLoadCmd->NumSections)
        return;
    } else if (LCI.Command.Type == macho::LCT_Segment64) {
      InMemoryStruct<macho::Segment64LoadCommand> Segment64LoadCmd;
      MachOObj->ReadSegment64LoadCommand(LCI, Segment64LoadCmd);
      if (DRI.d.b < Segment64LoadCmd->NumSections)
        return;
    }

    DRI.d.a++;
    DRI.d.b = 0;
  }
}

error_code MachOObjectFile::getSectionNext(DataRefImpl DRI,
                                           SectionRef &Result) const {
  DRI.d.b++;
  moveToNextSection(DRI);
  Result = SectionRef(DRI, this);
  return object_error::success;
}

void
MachOObjectFile::getSection(DataRefImpl DRI,
                            InMemoryStruct<macho::Section> &Res) const {
  InMemoryStruct<macho::SegmentLoadCommand> SLC;
  LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
  MachOObj->ReadSegmentLoadCommand(LCI, SLC);
  MachOObj->ReadSection(LCI, DRI.d.b, Res);
}

std::size_t MachOObjectFile::getSectionIndex(DataRefImpl Sec) const {
  SectionList::const_iterator loc =
    std::find(Sections.begin(), Sections.end(), Sec);
  assert(loc != Sections.end() && "Sec is not a valid section!");
  return std::distance(Sections.begin(), loc);
}

void
MachOObjectFile::getSection64(DataRefImpl DRI,
                            InMemoryStruct<macho::Section64> &Res) const {
  InMemoryStruct<macho::Segment64LoadCommand> SLC;
  LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
  MachOObj->ReadSegment64LoadCommand(LCI, SLC);
  MachOObj->ReadSection64(LCI, DRI.d.b, Res);
}

static bool is64BitLoadCommand(const MachOObject *MachOObj, DataRefImpl DRI) {
  LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
  if (LCI.Command.Type == macho::LCT_Segment64)
    return true;
  assert(LCI.Command.Type == macho::LCT_Segment && "Unexpected Type.");
  return false;
}

error_code MachOObjectFile::getSectionName(DataRefImpl DRI,
                                           StringRef &Result) const {
  // FIXME: thread safety.
  static char result[34];
  if (is64BitLoadCommand(MachOObj, DRI)) {
    InMemoryStruct<macho::Segment64LoadCommand> SLC;
    LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
    MachOObj->ReadSegment64LoadCommand(LCI, SLC);
    InMemoryStruct<macho::Section64> Sect;
    MachOObj->ReadSection64(LCI, DRI.d.b, Sect);

    strcpy(result, Sect->SegmentName);
    strcat(result, ",");
    strcat(result, Sect->Name);
  } else {
    InMemoryStruct<macho::SegmentLoadCommand> SLC;
    LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
    MachOObj->ReadSegmentLoadCommand(LCI, SLC);
    InMemoryStruct<macho::Section> Sect;
    MachOObj->ReadSection(LCI, DRI.d.b, Sect);

    strcpy(result, Sect->SegmentName);
    strcat(result, ",");
    strcat(result, Sect->Name);
  }
  Result = StringRef(result);
  return object_error::success;
}

error_code MachOObjectFile::getSectionAddress(DataRefImpl DRI,
                                              uint64_t &Result) const {
  if (is64BitLoadCommand(MachOObj, DRI)) {
    InMemoryStruct<macho::Section64> Sect;
    getSection64(DRI, Sect);
    Result = Sect->Address;
  } else {
    InMemoryStruct<macho::Section> Sect;
    getSection(DRI, Sect);
    Result = Sect->Address;
  }
  return object_error::success;
}

error_code MachOObjectFile::getSectionSize(DataRefImpl DRI,
                                           uint64_t &Result) const {
  if (is64BitLoadCommand(MachOObj, DRI)) {
    InMemoryStruct<macho::Section64> Sect;
    getSection64(DRI, Sect);
    Result = Sect->Size;
  } else {
    InMemoryStruct<macho::Section> Sect;
    getSection(DRI, Sect);
    Result = Sect->Size;
  }
  return object_error::success;
}

error_code MachOObjectFile::getSectionContents(DataRefImpl DRI,
                                               StringRef &Result) const {
  if (is64BitLoadCommand(MachOObj, DRI)) {
    InMemoryStruct<macho::Section64> Sect;
    getSection64(DRI, Sect);
    Result = MachOObj->getData(Sect->Offset, Sect->Size);
  } else {
    InMemoryStruct<macho::Section> Sect;
    getSection(DRI, Sect);
    Result = MachOObj->getData(Sect->Offset, Sect->Size);
  }
  return object_error::success;
}

error_code MachOObjectFile::getSectionAlignment(DataRefImpl DRI,
                                                uint64_t &Result) const {
  if (is64BitLoadCommand(MachOObj, DRI)) {
    InMemoryStruct<macho::Section64> Sect;
    getSection64(DRI, Sect);
    Result = uint64_t(1) << Sect->Align;
  } else {
    InMemoryStruct<macho::Section> Sect;
    getSection(DRI, Sect);
    Result = uint64_t(1) << Sect->Align;
  }
  return object_error::success;
}

error_code MachOObjectFile::isSectionText(DataRefImpl DRI,
                                          bool &Result) const {
  if (is64BitLoadCommand(MachOObj, DRI)) {
    InMemoryStruct<macho::Section64> Sect;
    getSection64(DRI, Sect);
    Result = !strcmp(Sect->Name, "__text");
  } else {
    InMemoryStruct<macho::Section> Sect;
    getSection(DRI, Sect);
    Result = !strcmp(Sect->Name, "__text");
  }
  return object_error::success;
}

error_code MachOObjectFile::isSectionData(DataRefImpl DRI,
                                          bool &Result) const {
  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

error_code MachOObjectFile::isSectionBSS(DataRefImpl DRI,
                                         bool &Result) const {
  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

error_code MachOObjectFile::isSectionRequiredForExecution(DataRefImpl Sec,
                                                          bool &Result) const {
  // FIXME: Unimplemented.
  Result = true;
  return object_error::success;
}

error_code MachOObjectFile::isSectionVirtual(DataRefImpl Sec,
                                             bool &Result) const {
  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

error_code MachOObjectFile::isSectionZeroInit(DataRefImpl DRI,
                                              bool &Result) const {
  if (MachOObj->is64Bit()) {
    InMemoryStruct<macho::Section64> Sect;
    getSection64(DRI, Sect);
    unsigned SectionType = Sect->Flags & MachO::SectionFlagMaskSectionType;
    Result = (SectionType == MachO::SectionTypeZeroFill ||
              SectionType == MachO::SectionTypeZeroFillLarge);
  } else {
    InMemoryStruct<macho::Section> Sect;
    getSection(DRI, Sect);
    unsigned SectionType = Sect->Flags & MachO::SectionFlagMaskSectionType;
    Result = (SectionType == MachO::SectionTypeZeroFill ||
              SectionType == MachO::SectionTypeZeroFillLarge);
  }

  return object_error::success;
}

error_code MachOObjectFile::isSectionReadOnlyData(DataRefImpl Sec,
                                                  bool &Result) const {
  // Consider using the code from isSectionText to look for __const sections.
  // Alternately, emit S_ATTR_PURE_INSTRUCTIONS and/or S_ATTR_SOME_INSTRUCTIONS
  // to use section attributes to distinguish code from data.

  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

error_code MachOObjectFile::sectionContainsSymbol(DataRefImpl Sec,
                                                  DataRefImpl Symb,
                                                  bool &Result) const {
  SymbolRef::Type ST;
  getSymbolType(Symb, ST);
  if (ST == SymbolRef::ST_Unknown) {
    Result = false;
    return object_error::success;
  }

  uint64_t SectBegin, SectEnd;
  getSectionAddress(Sec, SectBegin);
  getSectionSize(Sec, SectEnd);
  SectEnd += SectBegin;

  if (MachOObj->is64Bit()) {
    InMemoryStruct<macho::Symbol64TableEntry> Entry;
    getSymbol64TableEntry(Symb, Entry);
    uint64_t SymAddr= Entry->Value;
    Result = (SymAddr >= SectBegin) && (SymAddr < SectEnd);
  } else {
    InMemoryStruct<macho::SymbolTableEntry> Entry;
    getSymbolTableEntry(Symb, Entry);
    uint64_t SymAddr= Entry->Value;
    Result = (SymAddr >= SectBegin) && (SymAddr < SectEnd);
  }

  return object_error::success;
}

relocation_iterator MachOObjectFile::getSectionRelBegin(DataRefImpl Sec) const {
  DataRefImpl ret;
  ret.d.b = getSectionIndex(Sec);
  return relocation_iterator(RelocationRef(ret, this));
}
relocation_iterator MachOObjectFile::getSectionRelEnd(DataRefImpl Sec) const {
  uint32_t last_reloc;
  if (is64BitLoadCommand(MachOObj, Sec)) {
    InMemoryStruct<macho::Section64> Sect;
    getSection64(Sec, Sect);
    last_reloc = Sect->NumRelocationTableEntries;
  } else {
    InMemoryStruct<macho::Section> Sect;
    getSection(Sec, Sect);
    last_reloc = Sect->NumRelocationTableEntries;
  }
  DataRefImpl ret;
  ret.d.a = last_reloc;
  ret.d.b = getSectionIndex(Sec);
  return relocation_iterator(RelocationRef(ret, this));
}

section_iterator MachOObjectFile::begin_sections() const {
  DataRefImpl DRI;
  moveToNextSection(DRI);
  return section_iterator(SectionRef(DRI, this));
}

section_iterator MachOObjectFile::end_sections() const {
  DataRefImpl DRI;
  DRI.d.a = MachOObj->getHeader().NumLoadCommands;
  return section_iterator(SectionRef(DRI, this));
}

/*===-- Relocations -------------------------------------------------------===*/

void MachOObjectFile::
getRelocation(DataRefImpl Rel,
              InMemoryStruct<macho::RelocationEntry> &Res) const {
  uint32_t relOffset;
  if (MachOObj->is64Bit()) {
    InMemoryStruct<macho::Section64> Sect;
    getSection64(Sections[Rel.d.b], Sect);
    relOffset = Sect->RelocationTableOffset;
  } else {
    InMemoryStruct<macho::Section> Sect;
    getSection(Sections[Rel.d.b], Sect);
    relOffset = Sect->RelocationTableOffset;
  }
  MachOObj->ReadRelocationEntry(relOffset, Rel.d.a, Res);
}
error_code MachOObjectFile::getRelocationNext(DataRefImpl Rel,
                                              RelocationRef &Res) const {
  ++Rel.d.a;
  Res = RelocationRef(Rel, this);
  return object_error::success;
}
error_code MachOObjectFile::getRelocationAddress(DataRefImpl Rel,
                                                 uint64_t &Res) const {
  const uint8_t* sectAddress = 0;
  if (MachOObj->is64Bit()) {
    InMemoryStruct<macho::Section64> Sect;
    getSection64(Sections[Rel.d.b], Sect);
    sectAddress += Sect->Address;
  } else {
    InMemoryStruct<macho::Section> Sect;
    getSection(Sections[Rel.d.b], Sect);
    sectAddress += Sect->Address;
  }
  InMemoryStruct<macho::RelocationEntry> RE;
  getRelocation(Rel, RE);

  unsigned Arch = getArch();
  bool isScattered = (Arch != Triple::x86_64) &&
                     (RE->Word0 & macho::RF_Scattered);
  uint64_t RelAddr = 0;
  if (isScattered)
    RelAddr = RE->Word0 & 0xFFFFFF;
  else
    RelAddr = RE->Word0;

  Res = reinterpret_cast<uintptr_t>(sectAddress + RelAddr);
  return object_error::success;
}
error_code MachOObjectFile::getRelocationOffset(DataRefImpl Rel,
                                                uint64_t &Res) const {
  InMemoryStruct<macho::RelocationEntry> RE;
  getRelocation(Rel, RE);

  unsigned Arch = getArch();
  bool isScattered = (Arch != Triple::x86_64) &&
                     (RE->Word0 & macho::RF_Scattered);
  if (isScattered)
    Res = RE->Word0 & 0xFFFFFF;
  else
    Res = RE->Word0;
  return object_error::success;
}
error_code MachOObjectFile::getRelocationSymbol(DataRefImpl Rel,
                                                SymbolRef &Res) const {
  InMemoryStruct<macho::RelocationEntry> RE;
  getRelocation(Rel, RE);
  uint32_t SymbolIdx = RE->Word1 & 0xffffff;
  bool isExtern = (RE->Word1 >> 27) & 1;

  DataRefImpl Sym;
  moveToNextSymbol(Sym);
  if (isExtern) {
    for (unsigned i = 0; i < SymbolIdx; i++) {
      Sym.d.b++;
      moveToNextSymbol(Sym);
      assert(Sym.d.a < MachOObj->getHeader().NumLoadCommands &&
             "Relocation symbol index out of range!");
    }
  }
  Res = SymbolRef(Sym, this);
  return object_error::success;
}
error_code MachOObjectFile::getRelocationType(DataRefImpl Rel,
                                              uint64_t &Res) const {
  InMemoryStruct<macho::RelocationEntry> RE;
  getRelocation(Rel, RE);
  Res = RE->Word0;
  Res <<= 32;
  Res |= RE->Word1;
  return object_error::success;
}
error_code MachOObjectFile::getRelocationTypeName(DataRefImpl Rel,
                                          SmallVectorImpl<char> &Result) const {
  // TODO: Support scattered relocations.
  StringRef res;
  InMemoryStruct<macho::RelocationEntry> RE;
  getRelocation(Rel, RE);

  unsigned Arch = getArch();
  bool isScattered = (Arch != Triple::x86_64) &&
                     (RE->Word0 & macho::RF_Scattered);

  unsigned r_type;
  if (isScattered)
    r_type = (RE->Word0 >> 24) & 0xF;
  else
    r_type = (RE->Word1 >> 28) & 0xF;

  switch (Arch) {
    case Triple::x86: {
      static const char *const Table[] =  {
        "GENERIC_RELOC_VANILLA",
        "GENERIC_RELOC_PAIR",
        "GENERIC_RELOC_SECTDIFF",
        "GENERIC_RELOC_PB_LA_PTR",
        "GENERIC_RELOC_LOCAL_SECTDIFF",
        "GENERIC_RELOC_TLV" };

      if (r_type > 6)
        res = "Unknown";
      else
        res = Table[r_type];
      break;
    }
    case Triple::x86_64: {
      static const char *const Table[] =  {
        "X86_64_RELOC_UNSIGNED",
        "X86_64_RELOC_SIGNED",
        "X86_64_RELOC_BRANCH",
        "X86_64_RELOC_GOT_LOAD",
        "X86_64_RELOC_GOT",
        "X86_64_RELOC_SUBTRACTOR",
        "X86_64_RELOC_SIGNED_1",
        "X86_64_RELOC_SIGNED_2",
        "X86_64_RELOC_SIGNED_4",
        "X86_64_RELOC_TLV" };

      if (r_type > 9)
        res = "Unknown";
      else
        res = Table[r_type];
      break;
    }
    case Triple::arm: {
      static const char *const Table[] =  {
        "ARM_RELOC_VANILLA",
        "ARM_RELOC_PAIR",
        "ARM_RELOC_SECTDIFF",
        "ARM_RELOC_LOCAL_SECTDIFF",
        "ARM_RELOC_PB_LA_PTR",
        "ARM_RELOC_BR24",
        "ARM_THUMB_RELOC_BR22",
        "ARM_THUMB_32BIT_BRANCH",
        "ARM_RELOC_HALF",
        "ARM_RELOC_HALF_SECTDIFF" };

      if (r_type > 9)
        res = "Unknown";
      else
        res = Table[r_type];
      break;
    }
    case Triple::ppc: {
      static const char *const Table[] =  {
        "PPC_RELOC_VANILLA",
        "PPC_RELOC_PAIR",
        "PPC_RELOC_BR14",
        "PPC_RELOC_BR24",
        "PPC_RELOC_HI16",
        "PPC_RELOC_LO16",
        "PPC_RELOC_HA16",
        "PPC_RELOC_LO14",
        "PPC_RELOC_SECTDIFF",
        "PPC_RELOC_PB_LA_PTR",
        "PPC_RELOC_HI16_SECTDIFF",
        "PPC_RELOC_LO16_SECTDIFF",
        "PPC_RELOC_HA16_SECTDIFF",
        "PPC_RELOC_JBSR",
        "PPC_RELOC_LO14_SECTDIFF",
        "PPC_RELOC_LOCAL_SECTDIFF" };

      res = Table[r_type];
      break;
    }
    case Triple::UnknownArch:
      res = "Unknown";
      break;
  }
  Result.append(res.begin(), res.end());
  return object_error::success;
}
error_code MachOObjectFile::getRelocationAdditionalInfo(DataRefImpl Rel,
                                                        int64_t &Res) const {
  InMemoryStruct<macho::RelocationEntry> RE;
  getRelocation(Rel, RE);
  bool isExtern = (RE->Word1 >> 27) & 1;
  Res = 0;
  if (!isExtern) {
    const uint8_t* sectAddress = base();
    if (MachOObj->is64Bit()) {
      InMemoryStruct<macho::Section64> Sect;
      getSection64(Sections[Rel.d.b], Sect);
      sectAddress += Sect->Offset;
    } else {
      InMemoryStruct<macho::Section> Sect;
      getSection(Sections[Rel.d.b], Sect);
      sectAddress += Sect->Offset;
    }
    Res = reinterpret_cast<uintptr_t>(sectAddress);
  }
  return object_error::success;
}

// Helper to advance a section or symbol iterator multiple increments at a time.
template<class T>
error_code advance(T &it, size_t Val) {
  error_code ec;
  while (Val--) {
    it.increment(ec);
  }
  return ec;
}

template<class T>
void advanceTo(T &it, size_t Val) {
  if (error_code ec = advance(it, Val))
    report_fatal_error(ec.message());
}

void MachOObjectFile::printRelocationTargetName(
                                     InMemoryStruct<macho::RelocationEntry>& RE,
                                     raw_string_ostream &fmt) const {
  unsigned Arch = getArch();
  bool isScattered = (Arch != Triple::x86_64) &&
                     (RE->Word0 & macho::RF_Scattered);

  // Target of a scattered relocation is an address.  In the interest of
  // generating pretty output, scan through the symbol table looking for a
  // symbol that aligns with that address.  If we find one, print it.
  // Otherwise, we just print the hex address of the target.
  if (isScattered) {
    uint32_t Val = RE->Word1;

    error_code ec;
    for (symbol_iterator SI = begin_symbols(), SE = end_symbols(); SI != SE;
        SI.increment(ec)) {
      if (ec) report_fatal_error(ec.message());

      uint64_t Addr;
      StringRef Name;

      if ((ec = SI->getAddress(Addr)))
        report_fatal_error(ec.message());
      if (Addr != Val) continue;
      if ((ec = SI->getName(Name)))
        report_fatal_error(ec.message());
      fmt << Name;
      return;
    }

    // If we couldn't find a symbol that this relocation refers to, try
    // to find a section beginning instead.
    for (section_iterator SI = begin_sections(), SE = end_sections(); SI != SE;
         SI.increment(ec)) {
      if (ec) report_fatal_error(ec.message());

      uint64_t Addr;
      StringRef Name;

      if ((ec = SI->getAddress(Addr)))
        report_fatal_error(ec.message());
      if (Addr != Val) continue;
      if ((ec = SI->getName(Name)))
        report_fatal_error(ec.message());
      fmt << Name;
      return;
    }

    fmt << format("0x%x", Val);
    return;
  }

  StringRef S;
  bool isExtern = (RE->Word1 >> 27) & 1;
  uint32_t Val = RE->Word1 & 0xFFFFFF;

  if (isExtern) {
    symbol_iterator SI = begin_symbols();
    advanceTo(SI, Val);
    SI->getName(S);
  } else {
    section_iterator SI = begin_sections();
    advanceTo(SI, Val);
    SI->getName(S);
  }

  fmt << S;
}

error_code MachOObjectFile::getRelocationValueString(DataRefImpl Rel,
                                          SmallVectorImpl<char> &Result) const {
  InMemoryStruct<macho::RelocationEntry> RE;
  getRelocation(Rel, RE);

  unsigned Arch = getArch();
  bool isScattered = (Arch != Triple::x86_64) &&
                     (RE->Word0 & macho::RF_Scattered);

  std::string fmtbuf;
  raw_string_ostream fmt(fmtbuf);

  unsigned Type;
  if (isScattered)
    Type = (RE->Word0 >> 24) & 0xF;
  else
    Type = (RE->Word1 >> 28) & 0xF;

  bool isPCRel;
  if (isScattered)
    isPCRel = ((RE->Word0 >> 30) & 1);
  else
    isPCRel = ((RE->Word1 >> 24) & 1);

  // Determine any addends that should be displayed with the relocation.
  // These require decoding the relocation type, which is triple-specific.

  // X86_64 has entirely custom relocation types.
  if (Arch == Triple::x86_64) {
    bool isPCRel = ((RE->Word1 >> 24) & 1);

    switch (Type) {
      case macho::RIT_X86_64_GOTLoad:   // X86_64_RELOC_GOT_LOAD
      case macho::RIT_X86_64_GOT: {     // X86_64_RELOC_GOT
        printRelocationTargetName(RE, fmt);
        fmt << "@GOT";
        if (isPCRel) fmt << "PCREL";
        break;
      }
      case macho::RIT_X86_64_Subtractor: { // X86_64_RELOC_SUBTRACTOR
        InMemoryStruct<macho::RelocationEntry> RENext;
        DataRefImpl RelNext = Rel;
        RelNext.d.a++;
        getRelocation(RelNext, RENext);

        // X86_64_SUBTRACTOR must be followed by a relocation of type
        // X86_64_RELOC_UNSIGNED.
        // NOTE: Scattered relocations don't exist on x86_64.
        unsigned RType = (RENext->Word1 >> 28) & 0xF;
        if (RType != 0)
          report_fatal_error("Expected X86_64_RELOC_UNSIGNED after "
                             "X86_64_RELOC_SUBTRACTOR.");

        // The X86_64_RELOC_UNSIGNED contains the minuend symbol,
        // X86_64_SUBTRACTOR contains to the subtrahend.
        printRelocationTargetName(RENext, fmt);
        fmt << "-";
        printRelocationTargetName(RE, fmt);
      }
      case macho::RIT_X86_64_TLV:
        printRelocationTargetName(RE, fmt);
        fmt << "@TLV";
        if (isPCRel) fmt << "P";
        break;
      case macho::RIT_X86_64_Signed1: // X86_64_RELOC_SIGNED1
        printRelocationTargetName(RE, fmt);
        fmt << "-1";
        break;
      case macho::RIT_X86_64_Signed2: // X86_64_RELOC_SIGNED2
        printRelocationTargetName(RE, fmt);
        fmt << "-2";
        break;
      case macho::RIT_X86_64_Signed4: // X86_64_RELOC_SIGNED4
        printRelocationTargetName(RE, fmt);
        fmt << "-4";
        break;
      default:
        printRelocationTargetName(RE, fmt);
        break;
    }
  // X86 and ARM share some relocation types in common.
  } else if (Arch == Triple::x86 || Arch == Triple::arm) {
    // Generic relocation types...
    switch (Type) {
      case macho::RIT_Pair: // GENERIC_RELOC_PAIR - prints no info
        return object_error::success;
      case macho::RIT_Difference: { // GENERIC_RELOC_SECTDIFF
        InMemoryStruct<macho::RelocationEntry> RENext;
        DataRefImpl RelNext = Rel;
        RelNext.d.a++;
        getRelocation(RelNext, RENext);

        // X86 sect diff's must be followed by a relocation of type
        // GENERIC_RELOC_PAIR.
        bool isNextScattered = (Arch != Triple::x86_64) &&
                               (RENext->Word0 & macho::RF_Scattered);
        unsigned RType;
        if (isNextScattered)
          RType = (RENext->Word0 >> 24) & 0xF;
        else
          RType = (RENext->Word1 >> 28) & 0xF;
        if (RType != 1)
          report_fatal_error("Expected GENERIC_RELOC_PAIR after "
                             "GENERIC_RELOC_SECTDIFF.");

        printRelocationTargetName(RE, fmt);
        fmt << "-";
        printRelocationTargetName(RENext, fmt);
        break;
      }
    }

    if (Arch == Triple::x86) {
      // All X86 relocations that need special printing were already
      // handled in the generic code.
      switch (Type) {
        case macho::RIT_Generic_LocalDifference:{// GENERIC_RELOC_LOCAL_SECTDIFF
          InMemoryStruct<macho::RelocationEntry> RENext;
          DataRefImpl RelNext = Rel;
          RelNext.d.a++;
          getRelocation(RelNext, RENext);

          // X86 sect diff's must be followed by a relocation of type
          // GENERIC_RELOC_PAIR.
          bool isNextScattered = (Arch != Triple::x86_64) &&
                               (RENext->Word0 & macho::RF_Scattered);
          unsigned RType;
          if (isNextScattered)
            RType = (RENext->Word0 >> 24) & 0xF;
          else
            RType = (RENext->Word1 >> 28) & 0xF;
          if (RType != 1)
            report_fatal_error("Expected GENERIC_RELOC_PAIR after "
                               "GENERIC_RELOC_LOCAL_SECTDIFF.");

          printRelocationTargetName(RE, fmt);
          fmt << "-";
          printRelocationTargetName(RENext, fmt);
          break;
        }
        case macho::RIT_Generic_TLV: {
          printRelocationTargetName(RE, fmt);
          fmt << "@TLV";
          if (isPCRel) fmt << "P";
          break;
        }
        default:
          printRelocationTargetName(RE, fmt);
      }
    } else { // ARM-specific relocations
      switch (Type) {
        case macho::RIT_ARM_Half:             // ARM_RELOC_HALF
        case macho::RIT_ARM_HalfDifference: { // ARM_RELOC_HALF_SECTDIFF
          // Half relocations steal a bit from the length field to encode
          // whether this is an upper16 or a lower16 relocation.
          bool isUpper;
          if (isScattered)
            isUpper = (RE->Word0 >> 28) & 1;
          else
            isUpper = (RE->Word1 >> 25) & 1;

          if (isUpper)
            fmt << ":upper16:(";
          else
            fmt << ":lower16:(";
          printRelocationTargetName(RE, fmt);

          InMemoryStruct<macho::RelocationEntry> RENext;
          DataRefImpl RelNext = Rel;
          RelNext.d.a++;
          getRelocation(RelNext, RENext);

          // ARM half relocs must be followed by a relocation of type
          // ARM_RELOC_PAIR.
          bool isNextScattered = (Arch != Triple::x86_64) &&
                                 (RENext->Word0 & macho::RF_Scattered);
          unsigned RType;
          if (isNextScattered)
            RType = (RENext->Word0 >> 24) & 0xF;
          else
            RType = (RENext->Word1 >> 28) & 0xF;

          if (RType != 1)
            report_fatal_error("Expected ARM_RELOC_PAIR after "
                               "GENERIC_RELOC_HALF");

          // NOTE: The half of the target virtual address is stashed in the
          // address field of the secondary relocation, but we can't reverse
          // engineer the constant offset from it without decoding the movw/movt
          // instruction to find the other half in its immediate field.

          // ARM_RELOC_HALF_SECTDIFF encodes the second section in the
          // symbol/section pointer of the follow-on relocation.
          if (Type == macho::RIT_ARM_HalfDifference) {
            fmt << "-";
            printRelocationTargetName(RENext, fmt);
          }

          fmt << ")";
          break;
        }
        default: {
          printRelocationTargetName(RE, fmt);
        }
      }
    }
  } else
    printRelocationTargetName(RE, fmt);

  fmt.flush();
  Result.append(fmtbuf.begin(), fmtbuf.end());
  return object_error::success;
}

error_code MachOObjectFile::getRelocationHidden(DataRefImpl Rel,
                                                bool &Result) const {
  InMemoryStruct<macho::RelocationEntry> RE;
  getRelocation(Rel, RE);

  unsigned Arch = getArch();
  bool isScattered = (Arch != Triple::x86_64) &&
                     (RE->Word0 & macho::RF_Scattered);
  unsigned Type;
  if (isScattered)
    Type = (RE->Word0 >> 24) & 0xF;
  else
    Type = (RE->Word1 >> 28) & 0xF;

  Result = false;

  // On arches that use the generic relocations, GENERIC_RELOC_PAIR
  // is always hidden.
  if (Arch == Triple::x86 || Arch == Triple::arm) {
    if (Type == macho::RIT_Pair) Result = true;
  } else if (Arch == Triple::x86_64) {
    // On x86_64, X86_64_RELOC_UNSIGNED is hidden only when it follows
    // an X864_64_RELOC_SUBTRACTOR.
    if (Type == macho::RIT_X86_64_Unsigned && Rel.d.a > 0) {
      DataRefImpl RelPrev = Rel;
      RelPrev.d.a--;
      InMemoryStruct<macho::RelocationEntry> REPrev;
      getRelocation(RelPrev, REPrev);

      unsigned PrevType = (REPrev->Word1 >> 28) & 0xF;

      if (PrevType == macho::RIT_X86_64_Subtractor) Result = true;
    }
  }

  return object_error::success;
}

error_code MachOObjectFile::getLibraryNext(DataRefImpl LibData,
                                           LibraryRef &Res) const {
  report_fatal_error("Needed libraries unimplemented in MachOObjectFile");
}

error_code MachOObjectFile::getLibraryPath(DataRefImpl LibData,
                                           StringRef &Res) const {
  report_fatal_error("Needed libraries unimplemented in MachOObjectFile");
}


/*===-- Miscellaneous -----------------------------------------------------===*/

uint8_t MachOObjectFile::getBytesInAddress() const {
  return MachOObj->is64Bit() ? 8 : 4;
}

StringRef MachOObjectFile::getFileFormatName() const {
  if (!MachOObj->is64Bit()) {
    switch (MachOObj->getHeader().CPUType) {
    case llvm::MachO::CPUTypeI386:
      return "Mach-O 32-bit i386";
    case llvm::MachO::CPUTypeARM:
      return "Mach-O arm";
    case llvm::MachO::CPUTypePowerPC:
      return "Mach-O 32-bit ppc";
    default:
      assert((MachOObj->getHeader().CPUType & llvm::MachO::CPUArchABI64) == 0 &&
             "64-bit object file when we're not 64-bit?");
      return "Mach-O 32-bit unknown";
    }
  }

  switch (MachOObj->getHeader().CPUType) {
  case llvm::MachO::CPUTypeX86_64:
    return "Mach-O 64-bit x86-64";
  case llvm::MachO::CPUTypePowerPC64:
    return "Mach-O 64-bit ppc64";
  default:
    assert((MachOObj->getHeader().CPUType & llvm::MachO::CPUArchABI64) == 1 &&
           "32-bit object file when we're 64-bit?");
    return "Mach-O 64-bit unknown";
  }
}

unsigned MachOObjectFile::getArch() const {
  switch (MachOObj->getHeader().CPUType) {
  case llvm::MachO::CPUTypeI386:
    return Triple::x86;
  case llvm::MachO::CPUTypeX86_64:
    return Triple::x86_64;
  case llvm::MachO::CPUTypeARM:
    return Triple::arm;
  case llvm::MachO::CPUTypePowerPC:
    return Triple::ppc;
  case llvm::MachO::CPUTypePowerPC64:
    return Triple::ppc64;
  default:
    return Triple::UnknownArch;
  }
}

} // end namespace object
} // end namespace llvm
