//===- MachOWriter.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MachOWriter.h"
#include "MachOLayoutBuilder.h"
#include "Object.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Object/MachO.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>

namespace llvm {
namespace objcopy {
namespace macho {

size_t MachOWriter::headerSize() const {
  return Is64Bit ? sizeof(MachO::mach_header_64) : sizeof(MachO::mach_header);
}

size_t MachOWriter::loadCommandsSize() const { return O.Header.SizeOfCmds; }

size_t MachOWriter::symTableSize() const {
  return O.SymTable.Symbols.size() *
         (Is64Bit ? sizeof(MachO::nlist_64) : sizeof(MachO::nlist));
}

size_t MachOWriter::totalSize() const {
  // Going from tail to head and looking for an appropriate "anchor" to
  // calculate the total size assuming that all the offsets are either valid
  // ("true") or 0 (0 indicates that the corresponding part is missing).

  SmallVector<size_t, 7> Ends;
  if (O.SymTabCommandIndex) {
    const MachO::symtab_command &SymTabCommand =
        O.LoadCommands[*O.SymTabCommandIndex]
            .MachOLoadCommand.symtab_command_data;
    if (SymTabCommand.symoff)
      Ends.push_back(SymTabCommand.symoff + symTableSize());
    if (SymTabCommand.stroff)
      Ends.push_back(SymTabCommand.stroff + SymTabCommand.strsize);
  }
  if (O.DyLdInfoCommandIndex) {
    const MachO::dyld_info_command &DyLdInfoCommand =
        O.LoadCommands[*O.DyLdInfoCommandIndex]
            .MachOLoadCommand.dyld_info_command_data;
    if (DyLdInfoCommand.rebase_off) {
      assert((DyLdInfoCommand.rebase_size == O.Rebases.Opcodes.size()) &&
             "Incorrect rebase opcodes size");
      Ends.push_back(DyLdInfoCommand.rebase_off + DyLdInfoCommand.rebase_size);
    }
    if (DyLdInfoCommand.bind_off) {
      assert((DyLdInfoCommand.bind_size == O.Binds.Opcodes.size()) &&
             "Incorrect bind opcodes size");
      Ends.push_back(DyLdInfoCommand.bind_off + DyLdInfoCommand.bind_size);
    }
    if (DyLdInfoCommand.weak_bind_off) {
      assert((DyLdInfoCommand.weak_bind_size == O.WeakBinds.Opcodes.size()) &&
             "Incorrect weak bind opcodes size");
      Ends.push_back(DyLdInfoCommand.weak_bind_off +
                     DyLdInfoCommand.weak_bind_size);
    }
    if (DyLdInfoCommand.lazy_bind_off) {
      assert((DyLdInfoCommand.lazy_bind_size == O.LazyBinds.Opcodes.size()) &&
             "Incorrect lazy bind opcodes size");
      Ends.push_back(DyLdInfoCommand.lazy_bind_off +
                     DyLdInfoCommand.lazy_bind_size);
    }
    if (DyLdInfoCommand.export_off) {
      assert((DyLdInfoCommand.export_size == O.Exports.Trie.size()) &&
             "Incorrect trie size");
      Ends.push_back(DyLdInfoCommand.export_off + DyLdInfoCommand.export_size);
    }
  }

  if (O.DySymTabCommandIndex) {
    const MachO::dysymtab_command &DySymTabCommand =
        O.LoadCommands[*O.DySymTabCommandIndex]
            .MachOLoadCommand.dysymtab_command_data;

    if (DySymTabCommand.indirectsymoff)
      Ends.push_back(DySymTabCommand.indirectsymoff +
                     sizeof(uint32_t) * O.IndirectSymTable.Symbols.size());
  }

  if (O.CodeSignatureCommandIndex) {
    const MachO::linkedit_data_command &LinkEditDataCommand =
        O.LoadCommands[*O.CodeSignatureCommandIndex]
            .MachOLoadCommand.linkedit_data_command_data;
    if (LinkEditDataCommand.dataoff)
      Ends.push_back(LinkEditDataCommand.dataoff +
                     LinkEditDataCommand.datasize);
  }

  if (O.DataInCodeCommandIndex) {
    const MachO::linkedit_data_command &LinkEditDataCommand =
        O.LoadCommands[*O.DataInCodeCommandIndex]
            .MachOLoadCommand.linkedit_data_command_data;

    if (LinkEditDataCommand.dataoff)
      Ends.push_back(LinkEditDataCommand.dataoff +
                     LinkEditDataCommand.datasize);
  }

  if (O.FunctionStartsCommandIndex) {
    const MachO::linkedit_data_command &LinkEditDataCommand =
        O.LoadCommands[*O.FunctionStartsCommandIndex]
            .MachOLoadCommand.linkedit_data_command_data;

    if (LinkEditDataCommand.dataoff)
      Ends.push_back(LinkEditDataCommand.dataoff +
                     LinkEditDataCommand.datasize);
  }

  // Otherwise, use the last section / reloction.
  for (const LoadCommand &LC : O.LoadCommands)
    for (const std::unique_ptr<Section> &S : LC.Sections) {
      if (!S->hasValidOffset()) {
        assert((S->Offset == 0) && "Skipped section's offset must be zero");
        assert((S->isVirtualSection() || S->Size == 0) &&
               "Non-zero-fill sections with zero offset must have zero size");
        continue;
      }
      assert((S->Offset != 0) &&
             "Non-zero-fill section's offset cannot be zero");
      Ends.push_back(S->Offset + S->Size);
      if (S->RelOff)
        Ends.push_back(S->RelOff +
                       S->NReloc * sizeof(MachO::any_relocation_info));
    }

  if (!Ends.empty())
    return *std::max_element(Ends.begin(), Ends.end());

  // Otherwise, we have only Mach header and load commands.
  return headerSize() + loadCommandsSize();
}

void MachOWriter::writeHeader() {
  MachO::mach_header_64 Header;

  Header.magic = O.Header.Magic;
  Header.cputype = O.Header.CPUType;
  Header.cpusubtype = O.Header.CPUSubType;
  Header.filetype = O.Header.FileType;
  Header.ncmds = O.Header.NCmds;
  Header.sizeofcmds = O.Header.SizeOfCmds;
  Header.flags = O.Header.Flags;
  Header.reserved = O.Header.Reserved;

  if (IsLittleEndian != sys::IsLittleEndianHost)
    MachO::swapStruct(Header);

  auto HeaderSize =
      Is64Bit ? sizeof(MachO::mach_header_64) : sizeof(MachO::mach_header);
  memcpy(Buf->getBufferStart(), &Header, HeaderSize);
}

void MachOWriter::writeLoadCommands() {
  uint8_t *Begin =
      reinterpret_cast<uint8_t *>(Buf->getBufferStart()) + headerSize();
  for (const LoadCommand &LC : O.LoadCommands) {
    // Construct a load command.
    MachO::macho_load_command MLC = LC.MachOLoadCommand;
    switch (MLC.load_command_data.cmd) {
    case MachO::LC_SEGMENT:
      if (IsLittleEndian != sys::IsLittleEndianHost)
        MachO::swapStruct(MLC.segment_command_data);
      memcpy(Begin, &MLC.segment_command_data, sizeof(MachO::segment_command));
      Begin += sizeof(MachO::segment_command);

      for (const std::unique_ptr<Section> &Sec : LC.Sections)
        writeSectionInLoadCommand<MachO::section>(*Sec, Begin);
      continue;
    case MachO::LC_SEGMENT_64:
      if (IsLittleEndian != sys::IsLittleEndianHost)
        MachO::swapStruct(MLC.segment_command_64_data);
      memcpy(Begin, &MLC.segment_command_64_data,
             sizeof(MachO::segment_command_64));
      Begin += sizeof(MachO::segment_command_64);

      for (const std::unique_ptr<Section> &Sec : LC.Sections)
        writeSectionInLoadCommand<MachO::section_64>(*Sec, Begin);
      continue;
    }

#define HANDLE_LOAD_COMMAND(LCName, LCValue, LCStruct)                         \
  case MachO::LCName:                                                          \
    assert(sizeof(MachO::LCStruct) + LC.Payload.size() ==                      \
           MLC.load_command_data.cmdsize);                                     \
    if (IsLittleEndian != sys::IsLittleEndianHost)                             \
      MachO::swapStruct(MLC.LCStruct##_data);                                  \
    memcpy(Begin, &MLC.LCStruct##_data, sizeof(MachO::LCStruct));              \
    Begin += sizeof(MachO::LCStruct);                                          \
    if (!LC.Payload.empty())                                                   \
      memcpy(Begin, LC.Payload.data(), LC.Payload.size());                     \
    Begin += LC.Payload.size();                                                \
    break;

    // Copy the load command as it is.
    switch (MLC.load_command_data.cmd) {
    default:
      assert(sizeof(MachO::load_command) + LC.Payload.size() ==
             MLC.load_command_data.cmdsize);
      if (IsLittleEndian != sys::IsLittleEndianHost)
        MachO::swapStruct(MLC.load_command_data);
      memcpy(Begin, &MLC.load_command_data, sizeof(MachO::load_command));
      Begin += sizeof(MachO::load_command);
      if (!LC.Payload.empty())
        memcpy(Begin, LC.Payload.data(), LC.Payload.size());
      Begin += LC.Payload.size();
      break;
#include "llvm/BinaryFormat/MachO.def"
    }
  }
}

template <typename StructType>
void MachOWriter::writeSectionInLoadCommand(const Section &Sec, uint8_t *&Out) {
  StructType Temp;
  assert(Sec.Segname.size() <= sizeof(Temp.segname) && "too long segment name");
  assert(Sec.Sectname.size() <= sizeof(Temp.sectname) &&
         "too long section name");
  memset(&Temp, 0, sizeof(StructType));
  memcpy(Temp.segname, Sec.Segname.data(), Sec.Segname.size());
  memcpy(Temp.sectname, Sec.Sectname.data(), Sec.Sectname.size());
  Temp.addr = Sec.Addr;
  Temp.size = Sec.Size;
  Temp.offset = Sec.Offset;
  Temp.align = Sec.Align;
  Temp.reloff = Sec.RelOff;
  Temp.nreloc = Sec.NReloc;
  Temp.flags = Sec.Flags;
  Temp.reserved1 = Sec.Reserved1;
  Temp.reserved2 = Sec.Reserved2;

  if (IsLittleEndian != sys::IsLittleEndianHost)
    MachO::swapStruct(Temp);
  memcpy(Out, &Temp, sizeof(StructType));
  Out += sizeof(StructType);
}

void MachOWriter::writeSections() {
  for (const LoadCommand &LC : O.LoadCommands)
    for (const std::unique_ptr<Section> &Sec : LC.Sections) {
      if (!Sec->hasValidOffset()) {
        assert((Sec->Offset == 0) && "Skipped section's offset must be zero");
        assert((Sec->isVirtualSection() || Sec->Size == 0) &&
               "Non-zero-fill sections with zero offset must have zero size");
        continue;
      }

      assert(Sec->Offset && "Section offset can not be zero");
      assert((Sec->Size == Sec->Content.size()) && "Incorrect section size");
      memcpy(Buf->getBufferStart() + Sec->Offset, Sec->Content.data(),
             Sec->Content.size());
      for (size_t Index = 0; Index < Sec->Relocations.size(); ++Index) {
        RelocationInfo RelocInfo = Sec->Relocations[Index];
        if (!RelocInfo.Scattered) {
          const uint32_t SymbolNum = RelocInfo.Extern
                                         ? (*RelocInfo.Symbol)->Index
                                         : (*RelocInfo.Sec)->Index;
          RelocInfo.setPlainRelocationSymbolNum(SymbolNum, IsLittleEndian);
        }
        if (IsLittleEndian != sys::IsLittleEndianHost)
          MachO::swapStruct(
              reinterpret_cast<MachO::any_relocation_info &>(RelocInfo.Info));
        memcpy(Buf->getBufferStart() + Sec->RelOff +
                   Index * sizeof(MachO::any_relocation_info),
               &RelocInfo.Info, sizeof(RelocInfo.Info));
      }
    }
}

template <typename NListType>
void writeNListEntry(const SymbolEntry &SE, bool IsLittleEndian, char *&Out,
                     uint32_t Nstrx) {
  NListType ListEntry;
  ListEntry.n_strx = Nstrx;
  ListEntry.n_type = SE.n_type;
  ListEntry.n_sect = SE.n_sect;
  ListEntry.n_desc = SE.n_desc;
  ListEntry.n_value = SE.n_value;

  if (IsLittleEndian != sys::IsLittleEndianHost)
    MachO::swapStruct(ListEntry);
  memcpy(Out, reinterpret_cast<const char *>(&ListEntry), sizeof(NListType));
  Out += sizeof(NListType);
}

void MachOWriter::writeStringTable() {
  if (!O.SymTabCommandIndex)
    return;
  const MachO::symtab_command &SymTabCommand =
      O.LoadCommands[*O.SymTabCommandIndex]
          .MachOLoadCommand.symtab_command_data;

  uint8_t *StrTable = (uint8_t *)Buf->getBufferStart() + SymTabCommand.stroff;
  LayoutBuilder.getStringTableBuilder().write(StrTable);
}

void MachOWriter::writeSymbolTable() {
  if (!O.SymTabCommandIndex)
    return;
  const MachO::symtab_command &SymTabCommand =
      O.LoadCommands[*O.SymTabCommandIndex]
          .MachOLoadCommand.symtab_command_data;

  char *SymTable = (char *)Buf->getBufferStart() + SymTabCommand.symoff;
  for (auto Iter = O.SymTable.Symbols.begin(), End = O.SymTable.Symbols.end();
       Iter != End; Iter++) {
    SymbolEntry *Sym = Iter->get();
    uint32_t Nstrx = LayoutBuilder.getStringTableBuilder().getOffset(Sym->Name);

    if (Is64Bit)
      writeNListEntry<MachO::nlist_64>(*Sym, IsLittleEndian, SymTable, Nstrx);
    else
      writeNListEntry<MachO::nlist>(*Sym, IsLittleEndian, SymTable, Nstrx);
  }
}

void MachOWriter::writeRebaseInfo() {
  if (!O.DyLdInfoCommandIndex)
    return;
  const MachO::dyld_info_command &DyLdInfoCommand =
      O.LoadCommands[*O.DyLdInfoCommandIndex]
          .MachOLoadCommand.dyld_info_command_data;
  char *Out = (char *)Buf->getBufferStart() + DyLdInfoCommand.rebase_off;
  assert((DyLdInfoCommand.rebase_size == O.Rebases.Opcodes.size()) &&
         "Incorrect rebase opcodes size");
  memcpy(Out, O.Rebases.Opcodes.data(), O.Rebases.Opcodes.size());
}

void MachOWriter::writeBindInfo() {
  if (!O.DyLdInfoCommandIndex)
    return;
  const MachO::dyld_info_command &DyLdInfoCommand =
      O.LoadCommands[*O.DyLdInfoCommandIndex]
          .MachOLoadCommand.dyld_info_command_data;
  char *Out = (char *)Buf->getBufferStart() + DyLdInfoCommand.bind_off;
  assert((DyLdInfoCommand.bind_size == O.Binds.Opcodes.size()) &&
         "Incorrect bind opcodes size");
  memcpy(Out, O.Binds.Opcodes.data(), O.Binds.Opcodes.size());
}

void MachOWriter::writeWeakBindInfo() {
  if (!O.DyLdInfoCommandIndex)
    return;
  const MachO::dyld_info_command &DyLdInfoCommand =
      O.LoadCommands[*O.DyLdInfoCommandIndex]
          .MachOLoadCommand.dyld_info_command_data;
  char *Out = (char *)Buf->getBufferStart() + DyLdInfoCommand.weak_bind_off;
  assert((DyLdInfoCommand.weak_bind_size == O.WeakBinds.Opcodes.size()) &&
         "Incorrect weak bind opcodes size");
  memcpy(Out, O.WeakBinds.Opcodes.data(), O.WeakBinds.Opcodes.size());
}

void MachOWriter::writeLazyBindInfo() {
  if (!O.DyLdInfoCommandIndex)
    return;
  const MachO::dyld_info_command &DyLdInfoCommand =
      O.LoadCommands[*O.DyLdInfoCommandIndex]
          .MachOLoadCommand.dyld_info_command_data;
  char *Out = (char *)Buf->getBufferStart() + DyLdInfoCommand.lazy_bind_off;
  assert((DyLdInfoCommand.lazy_bind_size == O.LazyBinds.Opcodes.size()) &&
         "Incorrect lazy bind opcodes size");
  memcpy(Out, O.LazyBinds.Opcodes.data(), O.LazyBinds.Opcodes.size());
}

void MachOWriter::writeExportInfo() {
  if (!O.DyLdInfoCommandIndex)
    return;
  const MachO::dyld_info_command &DyLdInfoCommand =
      O.LoadCommands[*O.DyLdInfoCommandIndex]
          .MachOLoadCommand.dyld_info_command_data;
  char *Out = (char *)Buf->getBufferStart() + DyLdInfoCommand.export_off;
  assert((DyLdInfoCommand.export_size == O.Exports.Trie.size()) &&
         "Incorrect export trie size");
  memcpy(Out, O.Exports.Trie.data(), O.Exports.Trie.size());
}

void MachOWriter::writeIndirectSymbolTable() {
  if (!O.DySymTabCommandIndex)
    return;

  const MachO::dysymtab_command &DySymTabCommand =
      O.LoadCommands[*O.DySymTabCommandIndex]
          .MachOLoadCommand.dysymtab_command_data;

  uint32_t *Out =
      (uint32_t *)(Buf->getBufferStart() + DySymTabCommand.indirectsymoff);
  for (const IndirectSymbolEntry &Sym : O.IndirectSymTable.Symbols) {
    uint32_t Entry = (Sym.Symbol) ? (*Sym.Symbol)->Index : Sym.OriginalIndex;
    if (IsLittleEndian != sys::IsLittleEndianHost)
      sys::swapByteOrder(Entry);
    *Out++ = Entry;
  }
}

void MachOWriter::writeLinkData(Optional<size_t> LCIndex, const LinkData &LD) {
  if (!LCIndex)
    return;
  const MachO::linkedit_data_command &LinkEditDataCommand =
      O.LoadCommands[*LCIndex].MachOLoadCommand.linkedit_data_command_data;
  char *Out = (char *)Buf->getBufferStart() + LinkEditDataCommand.dataoff;
  assert((LinkEditDataCommand.datasize == LD.Data.size()) &&
         "Incorrect data size");
  memcpy(Out, LD.Data.data(), LD.Data.size());
}

void MachOWriter::writeCodeSignatureData() {
  return writeLinkData(O.CodeSignatureCommandIndex, O.CodeSignature);
}

void MachOWriter::writeDataInCodeData() {
  return writeLinkData(O.DataInCodeCommandIndex, O.DataInCode);
}

void MachOWriter::writeFunctionStartsData() {
  return writeLinkData(O.FunctionStartsCommandIndex, O.FunctionStarts);
}

void MachOWriter::writeTail() {
  typedef void (MachOWriter::*WriteHandlerType)(void);
  typedef std::pair<uint64_t, WriteHandlerType> WriteOperation;
  SmallVector<WriteOperation, 7> Queue;

  if (O.SymTabCommandIndex) {
    const MachO::symtab_command &SymTabCommand =
        O.LoadCommands[*O.SymTabCommandIndex]
            .MachOLoadCommand.symtab_command_data;
    if (SymTabCommand.symoff)
      Queue.push_back({SymTabCommand.symoff, &MachOWriter::writeSymbolTable});
    if (SymTabCommand.stroff)
      Queue.push_back({SymTabCommand.stroff, &MachOWriter::writeStringTable});
  }

  if (O.DyLdInfoCommandIndex) {
    const MachO::dyld_info_command &DyLdInfoCommand =
        O.LoadCommands[*O.DyLdInfoCommandIndex]
            .MachOLoadCommand.dyld_info_command_data;
    if (DyLdInfoCommand.rebase_off)
      Queue.push_back(
          {DyLdInfoCommand.rebase_off, &MachOWriter::writeRebaseInfo});
    if (DyLdInfoCommand.bind_off)
      Queue.push_back({DyLdInfoCommand.bind_off, &MachOWriter::writeBindInfo});
    if (DyLdInfoCommand.weak_bind_off)
      Queue.push_back(
          {DyLdInfoCommand.weak_bind_off, &MachOWriter::writeWeakBindInfo});
    if (DyLdInfoCommand.lazy_bind_off)
      Queue.push_back(
          {DyLdInfoCommand.lazy_bind_off, &MachOWriter::writeLazyBindInfo});
    if (DyLdInfoCommand.export_off)
      Queue.push_back(
          {DyLdInfoCommand.export_off, &MachOWriter::writeExportInfo});
  }

  if (O.DySymTabCommandIndex) {
    const MachO::dysymtab_command &DySymTabCommand =
        O.LoadCommands[*O.DySymTabCommandIndex]
            .MachOLoadCommand.dysymtab_command_data;

    if (DySymTabCommand.indirectsymoff)
      Queue.emplace_back(DySymTabCommand.indirectsymoff,
                         &MachOWriter::writeIndirectSymbolTable);
  }

  if (O.CodeSignatureCommandIndex) {
    const MachO::linkedit_data_command &LinkEditDataCommand =
        O.LoadCommands[*O.CodeSignatureCommandIndex]
            .MachOLoadCommand.linkedit_data_command_data;

    if (LinkEditDataCommand.dataoff)
      Queue.emplace_back(LinkEditDataCommand.dataoff,
                         &MachOWriter::writeCodeSignatureData);
  }

  if (O.DataInCodeCommandIndex) {
    const MachO::linkedit_data_command &LinkEditDataCommand =
        O.LoadCommands[*O.DataInCodeCommandIndex]
            .MachOLoadCommand.linkedit_data_command_data;

    if (LinkEditDataCommand.dataoff)
      Queue.emplace_back(LinkEditDataCommand.dataoff,
                         &MachOWriter::writeDataInCodeData);
  }

  if (O.FunctionStartsCommandIndex) {
    const MachO::linkedit_data_command &LinkEditDataCommand =
        O.LoadCommands[*O.FunctionStartsCommandIndex]
            .MachOLoadCommand.linkedit_data_command_data;

    if (LinkEditDataCommand.dataoff)
      Queue.emplace_back(LinkEditDataCommand.dataoff,
                         &MachOWriter::writeFunctionStartsData);
  }

  llvm::sort(Queue, [](const WriteOperation &LHS, const WriteOperation &RHS) {
    return LHS.first < RHS.first;
  });

  for (auto WriteOp : Queue)
    (this->*WriteOp.second)();
}

Error MachOWriter::finalize() { return LayoutBuilder.layout(); }

Error MachOWriter::write() {
  size_t TotalSize = totalSize();
  Buf = WritableMemoryBuffer::getNewMemBuffer(TotalSize);
  if (!Buf)
    return createStringError(errc::not_enough_memory,
                             "failed to allocate memory buffer of " +
                                 Twine::utohexstr(TotalSize) + " bytes");
  memset(Buf->getBufferStart(), 0, totalSize());
  writeHeader();
  writeLoadCommands();
  writeSections();
  writeTail();

  // TODO: Implement direct writing to the output stream (without intermediate
  // memory buffer Buf).
  Out.write(Buf->getBufferStart(), Buf->getBufferSize());
  return Error::success();
}

} // end namespace macho
} // end namespace objcopy
} // end namespace llvm
