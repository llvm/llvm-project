//===- MachOReader.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MachOReader.h"
#include "Object.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Object/MachO.h"
#include "llvm/Support/Errc.h"
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
Section constructSectionCommon(SectionType Sec, uint32_t Index) {
  StringRef SegName(Sec.segname, strnlen(Sec.segname, sizeof(Sec.segname)));
  StringRef SectName(Sec.sectname, strnlen(Sec.sectname, sizeof(Sec.sectname)));
  Section S(SegName, SectName);
  S.Index = Index;
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

template <typename SectionType>
Section constructSection(SectionType Sec, uint32_t Index);

template <> Section constructSection(MachO::section Sec, uint32_t Index) {
  return constructSectionCommon(Sec, Index);
}

template <> Section constructSection(MachO::section_64 Sec, uint32_t Index) {
  Section S = constructSectionCommon(Sec, Index);
  S.Reserved3 = Sec.reserved3;
  return S;
}

template <typename SectionType, typename SegmentType>
Expected<std::vector<std::unique_ptr<Section>>>
extractSections(const object::MachOObjectFile::LoadCommandInfo &LoadCmd,
                const object::MachOObjectFile &MachOObj,
                uint32_t &NextSectionIndex) {
  auto End = LoadCmd.Ptr + LoadCmd.C.cmdsize;
  const SectionType *Curr =
      reinterpret_cast<const SectionType *>(LoadCmd.Ptr + sizeof(SegmentType));
  std::vector<std::unique_ptr<Section>> Sections;
  for (; reinterpret_cast<const void *>(Curr) < End; Curr++) {
    if (MachOObj.isLittleEndian() != sys::IsLittleEndianHost) {
      SectionType Sec;
      memcpy((void *)&Sec, Curr, sizeof(SectionType));
      MachO::swapStruct(Sec);
      Sections.push_back(
          std::make_unique<Section>(constructSection(Sec, NextSectionIndex)));
    } else {
      Sections.push_back(
          std::make_unique<Section>(constructSection(*Curr, NextSectionIndex)));
    }

    Section &S = *Sections.back();

    Expected<object::SectionRef> SecRef =
        MachOObj.getSection(NextSectionIndex++);
    if (!SecRef)
      return SecRef.takeError();

    Expected<ArrayRef<uint8_t>> Data =
        MachOObj.getSectionContents(SecRef->getRawDataRefImpl());
    if (!Data)
      return Data.takeError();

    S.Content =
        StringRef(reinterpret_cast<const char *>(Data->data()), Data->size());

    S.Relocations.reserve(S.NReloc);
    for (auto RI = MachOObj.section_rel_begin(SecRef->getRawDataRefImpl()),
              RE = MachOObj.section_rel_end(SecRef->getRawDataRefImpl());
         RI != RE; ++RI) {
      RelocationInfo R;
      R.Symbol = nullptr; // We'll fill this field later.
      R.Info = MachOObj.getRelocation(RI->getRawDataRefImpl());
      R.Scattered = MachOObj.isRelocationScattered(R.Info);
      R.Extern = !R.Scattered && MachOObj.getPlainRelocationExternal(R.Info);
      S.Relocations.push_back(R);
    }

    assert(S.NReloc == S.Relocations.size() &&
           "Incorrect number of relocations");
  }
  return std::move(Sections);
}

Error MachOReader::readLoadCommands(Object &O) const {
  // For MachO sections indices start from 1.
  uint32_t NextSectionIndex = 1;
  for (auto LoadCmd : MachOObj.load_commands()) {
    LoadCommand LC;
    switch (LoadCmd.C.cmd) {
    case MachO::LC_CODE_SIGNATURE:
      O.CodeSignatureCommandIndex = O.LoadCommands.size();
      break;
    case MachO::LC_SEGMENT:
      if (Expected<std::vector<std::unique_ptr<Section>>> Sections =
              extractSections<MachO::section, MachO::segment_command>(
                  LoadCmd, MachOObj, NextSectionIndex))
        LC.Sections = std::move(*Sections);
      else
        return Sections.takeError();
      break;
    case MachO::LC_SEGMENT_64:
      if (Expected<std::vector<std::unique_ptr<Section>>> Sections =
              extractSections<MachO::section_64, MachO::segment_command_64>(
                  LoadCmd, MachOObj, NextSectionIndex))
        LC.Sections = std::move(*Sections);
      else
        return Sections.takeError();
      break;
    case MachO::LC_SYMTAB:
      O.SymTabCommandIndex = O.LoadCommands.size();
      break;
    case MachO::LC_DYSYMTAB:
      O.DySymTabCommandIndex = O.LoadCommands.size();
      break;
    case MachO::LC_DYLD_INFO:
    case MachO::LC_DYLD_INFO_ONLY:
      O.DyLdInfoCommandIndex = O.LoadCommands.size();
      break;
    case MachO::LC_DATA_IN_CODE:
      O.DataInCodeCommandIndex = O.LoadCommands.size();
      break;
    case MachO::LC_FUNCTION_STARTS:
      O.FunctionStartsCommandIndex = O.LoadCommands.size();
      break;
    }
#define HANDLE_LOAD_COMMAND(LCName, LCValue, LCStruct)                         \
  case MachO::LCName:                                                          \
    memcpy((void *)&(LC.MachOLoadCommand.LCStruct##_data), LoadCmd.Ptr,        \
           sizeof(MachO::LCStruct));                                           \
    if (MachOObj.isLittleEndian() != sys::IsLittleEndianHost)                  \
      MachO::swapStruct(LC.MachOLoadCommand.LCStruct##_data);                  \
    if (LoadCmd.C.cmdsize > sizeof(MachO::LCStruct))                           \
      LC.Payload = ArrayRef<uint8_t>(                                          \
          reinterpret_cast<uint8_t *>(const_cast<char *>(LoadCmd.Ptr)) +       \
              sizeof(MachO::LCStruct),                                         \
          LoadCmd.C.cmdsize - sizeof(MachO::LCStruct));                        \
    break;

    switch (LoadCmd.C.cmd) {
    default:
      memcpy((void *)&(LC.MachOLoadCommand.load_command_data), LoadCmd.Ptr,
             sizeof(MachO::load_command));
      if (MachOObj.isLittleEndian() != sys::IsLittleEndianHost)
        MachO::swapStruct(LC.MachOLoadCommand.load_command_data);
      if (LoadCmd.C.cmdsize > sizeof(MachO::load_command))
        LC.Payload = ArrayRef<uint8_t>(
            reinterpret_cast<uint8_t *>(const_cast<char *>(LoadCmd.Ptr)) +
                sizeof(MachO::load_command),
            LoadCmd.C.cmdsize - sizeof(MachO::load_command));
      break;
#include "llvm/BinaryFormat/MachO.def"
    }
    O.LoadCommands.push_back(std::move(LC));
  }
  return Error::success();
}

template <typename nlist_t>
SymbolEntry constructSymbolEntry(StringRef StrTable, const nlist_t &nlist) {
  assert(nlist.n_strx < StrTable.size() &&
         "n_strx exceeds the size of the string table");
  SymbolEntry SE;
  SE.Name = StringRef(StrTable.data() + nlist.n_strx).str();
  SE.n_type = nlist.n_type;
  SE.n_sect = nlist.n_sect;
  SE.n_desc = nlist.n_desc;
  SE.n_value = nlist.n_value;
  return SE;
}

void MachOReader::readSymbolTable(Object &O) const {
  StringRef StrTable = MachOObj.getStringTableData();
  for (auto Symbol : MachOObj.symbols()) {
    SymbolEntry SE =
        (MachOObj.is64Bit()
             ? constructSymbolEntry(StrTable, MachOObj.getSymbol64TableEntry(
                                                  Symbol.getRawDataRefImpl()))
             : constructSymbolEntry(StrTable, MachOObj.getSymbolTableEntry(
                                                  Symbol.getRawDataRefImpl())));

    O.SymTable.Symbols.push_back(std::make_unique<SymbolEntry>(SE));
  }
}

void MachOReader::setSymbolInRelocationInfo(Object &O) const {
  std::vector<const Section *> Sections;
  for (auto &LC : O.LoadCommands)
    for (std::unique_ptr<Section> &Sec : LC.Sections)
      Sections.push_back(Sec.get());

  for (LoadCommand &LC : O.LoadCommands)
    for (std::unique_ptr<Section> &Sec : LC.Sections)
      for (auto &Reloc : Sec->Relocations)
        if (!Reloc.Scattered) {
          const uint32_t SymbolNum =
              Reloc.getPlainRelocationSymbolNum(MachOObj.isLittleEndian());
          if (Reloc.Extern) {
            Reloc.Symbol = O.SymTable.getSymbolByIndex(SymbolNum);
          } else {
            // FIXME: Refactor error handling in MachOReader and report an error
            // if we encounter an invalid relocation.
            assert(SymbolNum >= 1 && SymbolNum <= Sections.size() &&
                   "Invalid section index.");
            Reloc.Sec = Sections[SymbolNum - 1];
          }
        }
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

void MachOReader::readLinkData(Object &O, Optional<size_t> LCIndex,
                               LinkData &LD) const {
  if (!LCIndex)
    return;
  const MachO::linkedit_data_command &LC =
      O.LoadCommands[*LCIndex].MachOLoadCommand.linkedit_data_command_data;
  LD.Data =
      arrayRefFromStringRef(MachOObj.getData().substr(LC.dataoff, LC.datasize));
}

void MachOReader::readCodeSignature(Object &O) const {
  return readLinkData(O, O.CodeSignatureCommandIndex, O.CodeSignature);
}

void MachOReader::readDataInCodeData(Object &O) const {
  return readLinkData(O, O.DataInCodeCommandIndex, O.DataInCode);
}

void MachOReader::readFunctionStartsData(Object &O) const {
  return readLinkData(O, O.FunctionStartsCommandIndex, O.FunctionStarts);
}

void MachOReader::readIndirectSymbolTable(Object &O) const {
  MachO::dysymtab_command DySymTab = MachOObj.getDysymtabLoadCommand();
  constexpr uint32_t AbsOrLocalMask =
      MachO::INDIRECT_SYMBOL_LOCAL | MachO::INDIRECT_SYMBOL_ABS;
  for (uint32_t i = 0; i < DySymTab.nindirectsyms; ++i) {
    uint32_t Index = MachOObj.getIndirectSymbolTableEntry(DySymTab, i);
    if ((Index & AbsOrLocalMask) != 0)
      O.IndirectSymTable.Symbols.emplace_back(Index, None);
    else
      O.IndirectSymTable.Symbols.emplace_back(
          Index, O.SymTable.getSymbolByIndex(Index));
  }
}

void MachOReader::readSwiftVersion(Object &O) const {
  struct ObjCImageInfo {
    uint32_t Version;
    uint32_t Flags;
  } ImageInfo;

  for (const LoadCommand &LC : O.LoadCommands)
    for (const std::unique_ptr<Section> &Sec : LC.Sections)
      if (Sec->Sectname == "__objc_imageinfo" &&
          (Sec->Segname == "__DATA" || Sec->Segname == "__DATA_CONST" ||
           Sec->Segname == "__DATA_DIRTY") &&
          Sec->Content.size() >= sizeof(ObjCImageInfo)) {
        memcpy(&ImageInfo, Sec->Content.data(), sizeof(ObjCImageInfo));
        if (MachOObj.isLittleEndian() != sys::IsLittleEndianHost) {
          sys::swapByteOrder(ImageInfo.Version);
          sys::swapByteOrder(ImageInfo.Flags);
        }
        O.SwiftVersion = (ImageInfo.Flags >> 8) & 0xff;
        return;
      }
}

Expected<std::unique_ptr<Object>> MachOReader::create() const {
  auto Obj = std::make_unique<Object>();
  readHeader(*Obj);
  if (Error E = readLoadCommands(*Obj))
    return std::move(E);
  readSymbolTable(*Obj);
  setSymbolInRelocationInfo(*Obj);
  readRebaseInfo(*Obj);
  readBindInfo(*Obj);
  readWeakBindInfo(*Obj);
  readLazyBindInfo(*Obj);
  readExportInfo(*Obj);
  readCodeSignature(*Obj);
  readDataInCodeData(*Obj);
  readFunctionStartsData(*Obj);
  readIndirectSymbolTable(*Obj);
  readSwiftVersion(*Obj);
  return std::move(Obj);
}

} // end namespace macho
} // end namespace objcopy
} // end namespace llvm
