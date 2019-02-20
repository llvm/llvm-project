//===- MachOReader.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MachOReader.h"
#include "../llvm-objcopy.h"
#include "Object.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Object/MachO.h"
#include <memory>

namespace llvm {
namespace objcopy {
namespace macho {

void MachOReader::readHeader(Object &O) const {
  O.Header.Magic = MachOObj.getHeader().magic;
  O.Header.CPUType = MachOObj.getHeader().cputype;
  O.Header.CPUSubType = MachOObj.getHeader().cpusubtype;
  O.Header.FileType = MachOObj.getHeader().filetype;
  O.Header.NCmds = MachOObj.getHeader().ncmds;
  O.Header.SizeOfCmds = MachOObj.getHeader().sizeofcmds;
  O.Header.Flags = MachOObj.getHeader().flags;
}

template <typename SectionType>
Section constructSectionCommon(SectionType Sec) {
  Section S;
  memcpy(S.Sectname, Sec.sectname, sizeof(Sec.sectname));
  memcpy(S.Segname, Sec.segname, sizeof(Sec.segname));
  S.Addr = Sec.addr;
  S.Size = Sec.size;
  S.Offset = Sec.offset;
  S.Align = Sec.align;
  S.RelOff = Sec.reloff;
  S.NReloc = Sec.nreloc;
  S.Flags = Sec.flags;
  S.Reserved1 = Sec.reserved1;
  S.Reserved2 = Sec.reserved2;
  S.Reserved3 = 0;
  return S;
}

template <typename SectionType> Section constructSection(SectionType Sec);

template <> Section constructSection(MachO::section Sec) {
  return constructSectionCommon(Sec);
}

template <> Section constructSection(MachO::section_64 Sec) {
  Section S = constructSectionCommon(Sec);
  S.Reserved3 = Sec.reserved3;
  return S;
}

// TODO: get rid of reportError and make MachOReader return Expected<> instead.
template <typename SectionType, typename SegmentType>
std::vector<Section>
extractSections(const object::MachOObjectFile::LoadCommandInfo &LoadCmd,
                const object::MachOObjectFile &MachOObj,
                size_t &NextSectionIndex) {
  auto End = LoadCmd.Ptr + LoadCmd.C.cmdsize;
  const SectionType *Curr =
      reinterpret_cast<const SectionType *>(LoadCmd.Ptr + sizeof(SegmentType));
  std::vector<Section> Sections;
  for (; reinterpret_cast<const void *>(Curr) < End; Curr++) {
    if (MachOObj.isLittleEndian() != sys::IsLittleEndianHost) {
      SectionType Sec;
      memcpy((void *)&Sec, Curr, sizeof(SectionType));
      MachO::swapStruct(Sec);
      Sections.push_back(constructSection(Sec));
    } else {
      Sections.push_back(constructSection(*Curr));
    }

    Section &S = Sections.back();

    StringRef SectName(S.Sectname);
    Expected<object::SectionRef> SecRef =
        MachOObj.getSection(NextSectionIndex++);
    if (!SecRef)
      reportError(MachOObj.getFileName(), SecRef.takeError());

    StringRef Content;
    if (auto EC =
            MachOObj.getSectionContents(SecRef->getRawDataRefImpl(), Content))
      reportError(MachOObj.getFileName(), std::move(EC));
    S.Content = Content;

    S.Relocations.reserve(S.NReloc);
    for (auto RI = MachOObj.section_rel_begin(SecRef->getRawDataRefImpl()),
              RE = MachOObj.section_rel_end(SecRef->getRawDataRefImpl());
         RI != RE; ++RI)
      S.Relocations.push_back(MachOObj.getRelocation(RI->getRawDataRefImpl()));
    assert(S.NReloc == S.Relocations.size() &&
           "Incorrect number of relocations");
  }
  return Sections;
}

void MachOReader::readLoadCommands(Object &O) const {
  // For MachO sections indices start from 1.
  size_t NextSectionIndex = 1;
  for (auto LoadCmd : MachOObj.load_commands()) {
    LoadCommand LC;
    switch (LoadCmd.C.cmd) {
    case MachO::LC_SEGMENT:
      LC.Sections = extractSections<MachO::section, MachO::segment_command>(
          LoadCmd, MachOObj, NextSectionIndex);
      break;
    case MachO::LC_SEGMENT_64:
      LC.Sections =
          extractSections<MachO::section_64, MachO::segment_command_64>(
              LoadCmd, MachOObj, NextSectionIndex);
      break;
    case MachO::LC_SYMTAB:
      O.SymTabCommandIndex = O.LoadCommands.size();
      break;
    case MachO::LC_DYLD_INFO:
    case MachO::LC_DYLD_INFO_ONLY:
      O.DyLdInfoCommandIndex = O.LoadCommands.size();
      break;
    }
#define HANDLE_LOAD_COMMAND(LCName, LCValue, LCStruct)                         \
  case MachO::LCName:                                                          \
    memcpy((void *)&(LC.MachOLoadCommand.LCStruct##_data), LoadCmd.Ptr,        \
           sizeof(MachO::LCStruct));                                           \
    if (MachOObj.isLittleEndian() != sys::IsLittleEndianHost)                  \
      MachO::swapStruct(LC.MachOLoadCommand.LCStruct##_data);                  \
    LC.Payload = ArrayRef<uint8_t>(                                            \
        reinterpret_cast<uint8_t *>(const_cast<char *>(LoadCmd.Ptr)) +         \
            sizeof(MachO::LCStruct),                                           \
        LoadCmd.C.cmdsize - sizeof(MachO::LCStruct));                          \
    break;

    switch (LoadCmd.C.cmd) {
    default:
      memcpy((void *)&(LC.MachOLoadCommand.load_command_data), LoadCmd.Ptr,
             sizeof(MachO::load_command));
      if (MachOObj.isLittleEndian() != sys::IsLittleEndianHost)
        MachO::swapStruct(LC.MachOLoadCommand.load_command_data);
      LC.Payload = ArrayRef<uint8_t>(
          reinterpret_cast<uint8_t *>(const_cast<char *>(LoadCmd.Ptr)) +
              sizeof(MachO::load_command),
          LoadCmd.C.cmdsize - sizeof(MachO::load_command));
      break;
#include "llvm/BinaryFormat/MachO.def"
    }
    O.LoadCommands.push_back(std::move(LC));
  }
}

template <typename nlist_t> NListEntry constructNameList(const nlist_t &nlist) {
  NListEntry NL;
  NL.n_strx = nlist.n_strx;
  NL.n_type = nlist.n_type;
  NL.n_sect = nlist.n_sect;
  NL.n_desc = nlist.n_desc;
  NL.n_value = nlist.n_value;
  return NL;
}

void MachOReader::readSymbolTable(Object &O) const {
  for (auto Symbol : MachOObj.symbols()) {
    NListEntry NLE =
        MachOObj.is64Bit()
            ? constructNameList<MachO::nlist_64>(
                  MachOObj.getSymbol64TableEntry(Symbol.getRawDataRefImpl()))
            : constructNameList<MachO::nlist>(
                  MachOObj.getSymbolTableEntry(Symbol.getRawDataRefImpl()));
    O.SymTable.NameList.push_back(NLE);
  }
}

void MachOReader::readStringTable(Object &O) const {
  StringRef Data = MachOObj.getStringTableData();
  SmallVector<StringRef, 10> Strs;
  Data.split(Strs, '\0');
  O.StrTable.Strings.reserve(Strs.size());
  for (auto S : Strs)
    O.StrTable.Strings.push_back(S.str());
}

void MachOReader::readRebaseInfo(Object &O) const {
  O.Rebases.Opcodes = MachOObj.getDyldInfoRebaseOpcodes();
}

void MachOReader::readBindInfo(Object &O) const {
  O.Binds.Opcodes = MachOObj.getDyldInfoBindOpcodes();
}

void MachOReader::readWeakBindInfo(Object &O) const {
  O.WeakBinds.Opcodes = MachOObj.getDyldInfoWeakBindOpcodes();
}

void MachOReader::readLazyBindInfo(Object &O) const {
  O.LazyBinds.Opcodes = MachOObj.getDyldInfoLazyBindOpcodes();
}

void MachOReader::readExportInfo(Object &O) const {
  O.Exports.Trie = MachOObj.getDyldInfoExportsTrie();
}

std::unique_ptr<Object> MachOReader::create() const {
  auto Obj = llvm::make_unique<Object>();
  readHeader(*Obj);
  readLoadCommands(*Obj);
  readSymbolTable(*Obj);
  readStringTable(*Obj);
  readRebaseInfo(*Obj);
  readBindInfo(*Obj);
  readWeakBindInfo(*Obj);
  readLazyBindInfo(*Obj);
  readExportInfo(*Obj);
  return Obj;
}

} // end namespace macho
} // end namespace objcopy
} // end namespace llvm
