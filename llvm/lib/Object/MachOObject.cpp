//===- MachOObject.cpp - Mach-O Object File Wrapper -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/MachOObject.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SwapByteOrder.h"

using namespace llvm;
using namespace llvm::object;

/* Translation Utilities */

template<typename T>
static void SwapValue(T &Value) {
  Value = sys::SwapByteOrder(Value);
}

template<typename T>
static void SwapStruct(T &Value);

template<typename T>
static void ReadInMemoryStruct(const MachOObject &MOO,
                               StringRef Buffer, uint64_t Base,
                               InMemoryStruct<T> &Res) {
  typedef T struct_type;
  uint64_t Size = sizeof(struct_type);

  // Check that the buffer contains the expected data.
  if (Base + Size >  Buffer.size()) {
    Res = 0;
    return;
  }

  // Check whether we can return a direct pointer.
  struct_type *Ptr = (struct_type *) (Buffer.data() + Base);
  if (!MOO.isSwappedEndian()) {
    Res = Ptr;
    return;
  }

  // Otherwise, copy the struct and translate the values.
  Res = *Ptr;
  SwapStruct(*Res);
}

/* *** */

MachOObject::MachOObject(MemoryBuffer *Buffer_, bool IsLittleEndian_,
                         bool Is64Bit_)
  : Buffer(Buffer_), IsLittleEndian(IsLittleEndian_), Is64Bit(Is64Bit_),
    IsSwappedEndian(IsLittleEndian != sys::isLittleEndianHost()),
    HasStringTable(false), LoadCommands(0), NumLoadedCommands(0) {
  // Load the common header.
  memcpy(&Header, Buffer->getBuffer().data(), sizeof(Header));
  if (IsSwappedEndian) {
    SwapValue(Header.Magic);
    SwapValue(Header.CPUType);
    SwapValue(Header.CPUSubtype);
    SwapValue(Header.FileType);
    SwapValue(Header.NumLoadCommands);
    SwapValue(Header.SizeOfLoadCommands);
    SwapValue(Header.Flags);
  }

  if (is64Bit()) {
    memcpy(&Header64Ext, Buffer->getBuffer().data() + sizeof(Header),
           sizeof(Header64Ext));
    if (IsSwappedEndian) {
      SwapValue(Header64Ext.Reserved);
    }
  }

  // Create the load command array if sane.
  if (getHeader().NumLoadCommands < (1 << 20))
    LoadCommands = new LoadCommandInfo[getHeader().NumLoadCommands];
}

MachOObject::~MachOObject() {
  delete [] LoadCommands;
}

MachOObject *MachOObject::LoadFromBuffer(MemoryBuffer *Buffer,
                                         std::string *ErrorStr) {
  // First, check the magic value and initialize the basic object info.
  bool IsLittleEndian = false, Is64Bit = false;
  StringRef Magic = Buffer->getBuffer().slice(0, 4);
  if (Magic == "\xFE\xED\xFA\xCE") {
  }  else if (Magic == "\xCE\xFA\xED\xFE") {
    IsLittleEndian = true;
  } else if (Magic == "\xFE\xED\xFA\xCF") {
    Is64Bit = true;
  } else if (Magic == "\xCF\xFA\xED\xFE") {
    IsLittleEndian = true;
    Is64Bit = true;
  } else {
    if (ErrorStr) *ErrorStr = "not a Mach object file (invalid magic)";
    return 0;
  }

  // Ensure that the at least the full header is present.
  unsigned HeaderSize = Is64Bit ? macho::Header64Size : macho::Header32Size;
  if (Buffer->getBufferSize() < HeaderSize) {
    if (ErrorStr) *ErrorStr = "not a Mach object file (invalid header)";
    return 0;
  }

  OwningPtr<MachOObject> Object(new MachOObject(Buffer, IsLittleEndian,
                                                Is64Bit));

  // Check for bogus number of load commands.
  if (Object->getHeader().NumLoadCommands >= (1 << 20)) {
    if (ErrorStr) *ErrorStr = "not a Mach object file (unreasonable header)";
    return 0;
  }

  if (ErrorStr) *ErrorStr = "";
  return Object.take();
}

StringRef MachOObject::getData(size_t Offset, size_t Size) const {
  return Buffer->getBuffer().substr(Offset,Size);
}

void MachOObject::RegisterStringTable(macho::SymtabLoadCommand &SLC) {
  HasStringTable = true;
  StringTable = Buffer->getBuffer().substr(SLC.StringTableOffset,
                                           SLC.StringTableSize);
}

const MachOObject::LoadCommandInfo &
MachOObject::getLoadCommandInfo(unsigned Index) const {
  assert(Index < getHeader().NumLoadCommands && "Invalid index!");

  // Load the command, if necessary.
  if (Index >= NumLoadedCommands) {
    uint64_t Offset;
    if (Index == 0) {
      Offset = getHeaderSize();
    } else {
      const LoadCommandInfo &Prev = getLoadCommandInfo(Index - 1);
      Offset = Prev.Offset + Prev.Command.Size;
    }

    LoadCommandInfo &Info = LoadCommands[Index];
    memcpy(&Info.Command, Buffer->getBuffer().data() + Offset,
           sizeof(macho::LoadCommand));
    if (IsSwappedEndian) {
      SwapValue(Info.Command.Type);
      SwapValue(Info.Command.Size);
    }
    Info.Offset = Offset;
    NumLoadedCommands = Index + 1;
  }

  return LoadCommands[Index];
}

template<>
void SwapStruct(macho::SegmentLoadCommand &Value) {
  SwapValue(Value.Type);
  SwapValue(Value.Size);
  SwapValue(Value.VMAddress);
  SwapValue(Value.VMSize);
  SwapValue(Value.FileOffset);
  SwapValue(Value.FileSize);
  SwapValue(Value.MaxVMProtection);
  SwapValue(Value.InitialVMProtection);
  SwapValue(Value.NumSections);
  SwapValue(Value.Flags);
}
void MachOObject::ReadSegmentLoadCommand(const LoadCommandInfo &LCI,
                         InMemoryStruct<macho::SegmentLoadCommand> &Res) const {
  ReadInMemoryStruct(*this, Buffer->getBuffer(), LCI.Offset, Res);
}

template<>
void SwapStruct(macho::Segment64LoadCommand &Value) {
  SwapValue(Value.Type);
  SwapValue(Value.Size);
  SwapValue(Value.VMAddress);
  SwapValue(Value.VMSize);
  SwapValue(Value.FileOffset);
  SwapValue(Value.FileSize);
  SwapValue(Value.MaxVMProtection);
  SwapValue(Value.InitialVMProtection);
  SwapValue(Value.NumSections);
  SwapValue(Value.Flags);
}
void MachOObject::ReadSegment64LoadCommand(const LoadCommandInfo &LCI,
                       InMemoryStruct<macho::Segment64LoadCommand> &Res) const {
  ReadInMemoryStruct(*this, Buffer->getBuffer(), LCI.Offset, Res);
}

template<>
void SwapStruct(macho::SymtabLoadCommand &Value) {
  SwapValue(Value.Type);
  SwapValue(Value.Size);
  SwapValue(Value.SymbolTableOffset);
  SwapValue(Value.NumSymbolTableEntries);
  SwapValue(Value.StringTableOffset);
  SwapValue(Value.StringTableSize);
}
void MachOObject::ReadSymtabLoadCommand(const LoadCommandInfo &LCI,
                          InMemoryStruct<macho::SymtabLoadCommand> &Res) const {
  ReadInMemoryStruct(*this, Buffer->getBuffer(), LCI.Offset, Res);
}

template<>
void SwapStruct(macho::DysymtabLoadCommand &Value) {
  SwapValue(Value.Type);
  SwapValue(Value.Size);
  SwapValue(Value.LocalSymbolsIndex);
  SwapValue(Value.NumLocalSymbols);
  SwapValue(Value.ExternalSymbolsIndex);
  SwapValue(Value.NumExternalSymbols);
  SwapValue(Value.UndefinedSymbolsIndex);
  SwapValue(Value.NumUndefinedSymbols);
  SwapValue(Value.TOCOffset);
  SwapValue(Value.NumTOCEntries);
  SwapValue(Value.ModuleTableOffset);
  SwapValue(Value.NumModuleTableEntries);
  SwapValue(Value.ReferenceSymbolTableOffset);
  SwapValue(Value.NumReferencedSymbolTableEntries);
  SwapValue(Value.IndirectSymbolTableOffset);
  SwapValue(Value.NumIndirectSymbolTableEntries);
  SwapValue(Value.ExternalRelocationTableOffset);
  SwapValue(Value.NumExternalRelocationTableEntries);
  SwapValue(Value.LocalRelocationTableOffset);
  SwapValue(Value.NumLocalRelocationTableEntries);
}
void MachOObject::ReadDysymtabLoadCommand(const LoadCommandInfo &LCI,
                        InMemoryStruct<macho::DysymtabLoadCommand> &Res) const {
  ReadInMemoryStruct(*this, Buffer->getBuffer(), LCI.Offset, Res);
}

template<>
void SwapStruct(macho::LinkeditDataLoadCommand &Value) {
  SwapValue(Value.Type);
  SwapValue(Value.Size);
  SwapValue(Value.DataOffset);
  SwapValue(Value.DataSize);
}
void MachOObject::ReadLinkeditDataLoadCommand(const LoadCommandInfo &LCI,
                    InMemoryStruct<macho::LinkeditDataLoadCommand> &Res) const {
  ReadInMemoryStruct(*this, Buffer->getBuffer(), LCI.Offset, Res);
}

template<>
void SwapStruct(macho::IndirectSymbolTableEntry &Value) {
  SwapValue(Value.Index);
}
void
MachOObject::ReadIndirectSymbolTableEntry(const macho::DysymtabLoadCommand &DLC,
                                          unsigned Index,
                   InMemoryStruct<macho::IndirectSymbolTableEntry> &Res) const {
  uint64_t Offset = (DLC.IndirectSymbolTableOffset +
                     Index * sizeof(macho::IndirectSymbolTableEntry));
  ReadInMemoryStruct(*this, Buffer->getBuffer(), Offset, Res);
}


template<>
void SwapStruct(macho::Section &Value) {
  SwapValue(Value.Address);
  SwapValue(Value.Size);
  SwapValue(Value.Offset);
  SwapValue(Value.Align);
  SwapValue(Value.RelocationTableOffset);
  SwapValue(Value.NumRelocationTableEntries);
  SwapValue(Value.Flags);
  SwapValue(Value.Reserved1);
  SwapValue(Value.Reserved2);
}
void MachOObject::ReadSection(const LoadCommandInfo &LCI,
                              unsigned Index,
                              InMemoryStruct<macho::Section> &Res) const {
  assert(LCI.Command.Type == macho::LCT_Segment &&
         "Unexpected load command info!");
  uint64_t Offset = (LCI.Offset + sizeof(macho::SegmentLoadCommand) +
                     Index * sizeof(macho::Section));
  ReadInMemoryStruct(*this, Buffer->getBuffer(), Offset, Res);
}

template<>
void SwapStruct(macho::Section64 &Value) {
  SwapValue(Value.Address);
  SwapValue(Value.Size);
  SwapValue(Value.Offset);
  SwapValue(Value.Align);
  SwapValue(Value.RelocationTableOffset);
  SwapValue(Value.NumRelocationTableEntries);
  SwapValue(Value.Flags);
  SwapValue(Value.Reserved1);
  SwapValue(Value.Reserved2);
  SwapValue(Value.Reserved3);
}
void MachOObject::ReadSection64(const LoadCommandInfo &LCI,
                                unsigned Index,
                                InMemoryStruct<macho::Section64> &Res) const {
  assert(LCI.Command.Type == macho::LCT_Segment64 &&
         "Unexpected load command info!");
  uint64_t Offset = (LCI.Offset + sizeof(macho::Segment64LoadCommand) +
                     Index * sizeof(macho::Section64));
  ReadInMemoryStruct(*this, Buffer->getBuffer(), Offset, Res);
}

template<>
void SwapStruct(macho::RelocationEntry &Value) {
  SwapValue(Value.Word0);
  SwapValue(Value.Word1);
}
void MachOObject::ReadRelocationEntry(uint64_t RelocationTableOffset,
                                      unsigned Index,
                            InMemoryStruct<macho::RelocationEntry> &Res) const {
  uint64_t Offset = (RelocationTableOffset +
                     Index * sizeof(macho::RelocationEntry));
  ReadInMemoryStruct(*this, Buffer->getBuffer(), Offset, Res);
}

template<>
void SwapStruct(macho::SymbolTableEntry &Value) {
  SwapValue(Value.StringIndex);
  SwapValue(Value.Flags);
  SwapValue(Value.Value);
}
void MachOObject::ReadSymbolTableEntry(uint64_t SymbolTableOffset,
                                       unsigned Index,
                           InMemoryStruct<macho::SymbolTableEntry> &Res) const {
  uint64_t Offset = (SymbolTableOffset +
                     Index * sizeof(macho::SymbolTableEntry));
  ReadInMemoryStruct(*this, Buffer->getBuffer(), Offset, Res);
}

template<>
void SwapStruct(macho::Symbol64TableEntry &Value) {
  SwapValue(Value.StringIndex);
  SwapValue(Value.Flags);
  SwapValue(Value.Value);
}
void MachOObject::ReadSymbol64TableEntry(uint64_t SymbolTableOffset,
                                       unsigned Index,
                         InMemoryStruct<macho::Symbol64TableEntry> &Res) const {
  uint64_t Offset = (SymbolTableOffset +
                     Index * sizeof(macho::Symbol64TableEntry));
  ReadInMemoryStruct(*this, Buffer->getBuffer(), Offset, Res);
}

template<>
void SwapStruct(macho::DataInCodeTableEntry &Value) {
  SwapValue(Value.Offset);
  SwapValue(Value.Length);
  SwapValue(Value.Kind);
}
void MachOObject::ReadDataInCodeTableEntry(uint64_t TableOffset,
                                           unsigned Index,
                       InMemoryStruct<macho::DataInCodeTableEntry> &Res) const {
  uint64_t Offset = (TableOffset +
                     Index * sizeof(macho::DataInCodeTableEntry));
  ReadInMemoryStruct(*this, Buffer->getBuffer(), Offset, Res);
}

void MachOObject::ReadULEB128s(uint64_t Index,
                               SmallVectorImpl<uint64_t> &Out) const {
  DataExtractor extractor(Buffer->getBuffer(), true, 0);

  uint32_t offset = Index;
  uint64_t data = 0;
  while (uint64_t delta = extractor.getULEB128(&offset)) {
    data += delta;
    Out.push_back(data);
  }
}

/* ** */
// Object Dumping Facilities
void MachOObject::dump() const { print(dbgs()); dbgs() << '\n'; }
void MachOObject::dumpHeader() const { printHeader(dbgs()); dbgs() << '\n'; }

void MachOObject::printHeader(raw_ostream &O) const {
  O << "('cputype', " << Header.CPUType << ")\n";
  O << "('cpusubtype', " << Header.CPUSubtype << ")\n";
  O << "('filetype', " << Header.FileType << ")\n";
  O << "('num_load_commands', " << Header.NumLoadCommands << ")\n";
  O << "('load_commands_size', " << Header.SizeOfLoadCommands << ")\n";
  O << "('flag', " << Header.Flags << ")\n";

  // Print extended header if 64-bit.
  if (is64Bit())
    O << "('reserved', " << Header64Ext.Reserved << ")\n";
}

void MachOObject::print(raw_ostream &O) const {
  O << "Header:\n";
  printHeader(O);
  O << "Load Commands:\n";

  O << "Buffer:\n";
}
