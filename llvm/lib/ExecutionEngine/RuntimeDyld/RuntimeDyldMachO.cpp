//===-- RuntimeDyldMachO.cpp - Run-time dynamic linker for MC-JIT -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the MC-JIT runtime dynamic linker.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "dyld"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"
#include "RuntimeDyldMachO.h"
using namespace llvm;
using namespace llvm::object;

namespace llvm {

void RuntimeDyldMachO::resolveRelocation(const SectionEntry &Section,
                                         uint64_t Offset,
                                         uint64_t Value,
                                         uint32_t Type,
                                         int64_t Addend) {
  uint8_t *LocalAddress = Section.Address + Offset;
  uint64_t FinalAddress = Section.LoadAddress + Offset;
  bool isPCRel = (Type >> 24) & 1;
  unsigned MachoType = (Type >> 28) & 0xf;
  unsigned Size = 1 << ((Type >> 25) & 3);

  DEBUG(dbgs() << "resolveRelocation LocalAddress: " 
        << format("%p", LocalAddress)
        << " FinalAddress: " << format("%p", FinalAddress)
        << " Value: " << format("%p", Value)
        << " Addend: " << Addend
        << " isPCRel: " << isPCRel
        << " MachoType: " << MachoType
        << " Size: " << Size
        << "\n");

  // This just dispatches to the proper target specific routine.
  switch (Arch) {
  default: llvm_unreachable("Unsupported CPU type!");
  case Triple::x86_64:
    resolveX86_64Relocation(LocalAddress,
                            FinalAddress,
                            (uintptr_t)Value,
                            isPCRel,
                            MachoType,
                            Size,
                            Addend);
    break;
  case Triple::x86:
    resolveI386Relocation(LocalAddress,
                          FinalAddress,
                          (uintptr_t)Value,
                          isPCRel,
                          MachoType,
                          Size,
                          Addend);
    break;
  case Triple::arm:    // Fall through.
  case Triple::thumb:
    resolveARMRelocation(LocalAddress,
                         FinalAddress,
                         (uintptr_t)Value,
                         isPCRel,
                         MachoType,
                         Size,
                         Addend);
    break;
  }
}

bool RuntimeDyldMachO::resolveI386Relocation(uint8_t *LocalAddress,
                                             uint64_t FinalAddress,
                                             uint64_t Value,
                                             bool isPCRel,
                                             unsigned Type,
                                             unsigned Size,
                                             int64_t Addend) {
  if (isPCRel)
    Value -= FinalAddress + 4; // see resolveX86_64Relocation

  switch (Type) {
  default:
    llvm_unreachable("Invalid relocation type!");
  case macho::RIT_Vanilla: {
    uint8_t *p = LocalAddress;
    uint64_t ValueToWrite = Value + Addend;
    for (unsigned i = 0; i < Size; ++i) {
      *p++ = (uint8_t)(ValueToWrite & 0xff);
      ValueToWrite >>= 8;
    }
  }
  case macho::RIT_Difference:
  case macho::RIT_Generic_LocalDifference:
  case macho::RIT_Generic_PreboundLazyPointer:
    return Error("Relocation type not implemented yet!");
  }
}

bool RuntimeDyldMachO::resolveX86_64Relocation(uint8_t *LocalAddress,
                                               uint64_t FinalAddress,
                                               uint64_t Value,
                                               bool isPCRel,
                                               unsigned Type,
                                               unsigned Size,
                                               int64_t Addend) {
  // If the relocation is PC-relative, the value to be encoded is the
  // pointer difference.
  if (isPCRel)
    // FIXME: It seems this value needs to be adjusted by 4 for an effective PC
    // address. Is that expected? Only for branches, perhaps?
    Value -= FinalAddress + 4;

  switch(Type) {
  default:
    llvm_unreachable("Invalid relocation type!");
  case macho::RIT_X86_64_Signed1:
  case macho::RIT_X86_64_Signed2:
  case macho::RIT_X86_64_Signed4:
  case macho::RIT_X86_64_Signed:
  case macho::RIT_X86_64_Unsigned:
  case macho::RIT_X86_64_Branch: {
    Value += Addend;
    // Mask in the target value a byte at a time (we don't have an alignment
    // guarantee for the target address, so this is safest).
    uint8_t *p = (uint8_t*)LocalAddress;
    for (unsigned i = 0; i < Size; ++i) {
      *p++ = (uint8_t)Value;
      Value >>= 8;
    }
    return false;
  }
  case macho::RIT_X86_64_GOTLoad:
  case macho::RIT_X86_64_GOT:
  case macho::RIT_X86_64_Subtractor:
  case macho::RIT_X86_64_TLV:
    return Error("Relocation type not implemented yet!");
  }
}

bool RuntimeDyldMachO::resolveARMRelocation(uint8_t *LocalAddress,
                                            uint64_t FinalAddress,
                                            uint64_t Value,
                                            bool isPCRel,
                                            unsigned Type,
                                            unsigned Size,
                                            int64_t Addend) {
  // If the relocation is PC-relative, the value to be encoded is the
  // pointer difference.
  if (isPCRel) {
    Value -= FinalAddress;
    // ARM PCRel relocations have an effective-PC offset of two instructions
    // (four bytes in Thumb mode, 8 bytes in ARM mode).
    // FIXME: For now, assume ARM mode.
    Value -= 8;
  }

  switch(Type) {
  default:
    llvm_unreachable("Invalid relocation type!");
  case macho::RIT_Vanilla: {
    // Mask in the target value a byte at a time (we don't have an alignment
    // guarantee for the target address, so this is safest).
    uint8_t *p = (uint8_t*)LocalAddress;
    for (unsigned i = 0; i < Size; ++i) {
      *p++ = (uint8_t)Value;
      Value >>= 8;
    }
    break;
  }
  case macho::RIT_ARM_Branch24Bit: {
    // Mask the value into the target address. We know instructions are
    // 32-bit aligned, so we can do it all at once.
    uint32_t *p = (uint32_t*)LocalAddress;
    // The low two bits of the value are not encoded.
    Value >>= 2;
    // Mask the value to 24 bits.
    Value &= 0xffffff;
    // FIXME: If the destination is a Thumb function (and the instruction
    // is a non-predicated BL instruction), we need to change it to a BLX
    // instruction instead.

    // Insert the value into the instruction.
    *p = (*p & ~0xffffff) | Value;
    break;
  }
  case macho::RIT_ARM_ThumbBranch22Bit:
  case macho::RIT_ARM_ThumbBranch32Bit:
  case macho::RIT_ARM_Half:
  case macho::RIT_ARM_HalfDifference:
  case macho::RIT_Pair:
  case macho::RIT_Difference:
  case macho::RIT_ARM_LocalDifference:
  case macho::RIT_ARM_PreboundLazyPointer:
    return Error("Relocation type not implemented yet!");
  }
  return false;
}

void RuntimeDyldMachO::processRelocationRef(const ObjRelocationInfo &Rel,
                                            ObjectImage &Obj,
                                            ObjSectionToIDMap &ObjSectionToID,
                                            const SymbolTableMap &Symbols,
                                            StubMap &Stubs) {

  uint32_t RelType = (uint32_t) (Rel.Type & 0xffffffffL);
  RelocationValueRef Value;
  SectionEntry &Section = Sections[Rel.SectionID];

  bool isExtern = (RelType >> 27) & 1;
  if (isExtern) {
    // Obtain the symbol name which is referenced in the relocation
    StringRef TargetName;
    const SymbolRef &Symbol = Rel.Symbol;
    Symbol.getName(TargetName);
    // First search for the symbol in the local symbol table
    SymbolTableMap::const_iterator lsi = Symbols.find(TargetName.data());
    if (lsi != Symbols.end()) {
      Value.SectionID = lsi->second.first;
      Value.Addend = lsi->second.second;
    } else {
      // Search for the symbol in the global symbol table
      SymbolTableMap::const_iterator gsi = GlobalSymbolTable.find(TargetName.data());
      if (gsi != GlobalSymbolTable.end()) {
        Value.SectionID = gsi->second.first;
        Value.Addend = gsi->second.second;
      } else
        Value.SymbolName = TargetName.data();
    }
  } else {
    error_code err;
    uint8_t sectionIndex = static_cast<uint8_t>(RelType & 0xFF);
    section_iterator si = Obj.begin_sections(),
                     se = Obj.end_sections();
    for (uint8_t i = 1; i < sectionIndex; i++) {
      error_code err;
      si.increment(err);
      if (si == se)
        break;
    }
    assert(si != se && "No section containing relocation!");
    Value.SectionID = findOrEmitSection(Obj, *si, true, ObjSectionToID);
    Value.Addend = 0;
    // FIXME: The size and type of the relocation determines if we can
    // encode an Addend in the target location itself, and if so, how many
    // bytes we should read in order to get it. We don't yet support doing
    // that, and just assuming it's sizeof(intptr_t) is blatantly wrong.
    //Value.Addend = *(const intptr_t *)Target;
    if (Value.Addend) {
      // The MachO addend is an offset from the current section.  We need it
      // to be an offset from the destination section
      Value.Addend += Section.ObjAddress - Sections[Value.SectionID].ObjAddress;
    }
  }

  if (Arch == Triple::arm && (RelType & 0xf) == macho::RIT_ARM_Branch24Bit) {
    // This is an ARM branch relocation, need to use a stub function.

    //  Look up for existing stub.
    StubMap::const_iterator i = Stubs.find(Value);
    if (i != Stubs.end())
      resolveRelocation(Section, Rel.Offset,
                        (uint64_t)Section.Address + i->second,
                        RelType, 0);
    else {
      // Create a new stub function.
      Stubs[Value] = Section.StubOffset;
      uint8_t *StubTargetAddr = createStubFunction(Section.Address +
                                                   Section.StubOffset);
      RelocationEntry RE(Rel.SectionID, StubTargetAddr - Section.Address,
                         macho::RIT_Vanilla, Value.Addend);
      if (Value.SymbolName)
        addRelocationForSymbol(RE, Value.SymbolName);
      else
        addRelocationForSection(RE, Value.SectionID);
      resolveRelocation(Section, Rel.Offset,
                        (uint64_t)Section.Address + Section.StubOffset,
                        RelType, 0);
      Section.StubOffset += getMaxStubSize();
    }
  } else {
    RelocationEntry RE(Rel.SectionID, Rel.Offset, RelType, Value.Addend);
    if (Value.SymbolName)
      addRelocationForSymbol(RE, Value.SymbolName);
    else
      addRelocationForSection(RE, Value.SectionID);
  }
}


bool RuntimeDyldMachO::isCompatibleFormat(
        const ObjectBuffer *InputBuffer) const {
  if (InputBuffer->getBufferSize() < 4)
    return false;
  StringRef Magic(InputBuffer->getBufferStart(), 4);
  if (Magic == "\xFE\xED\xFA\xCE") return true;
  if (Magic == "\xCE\xFA\xED\xFE") return true;
  if (Magic == "\xFE\xED\xFA\xCF") return true;
  if (Magic == "\xCF\xFA\xED\xFE") return true;
  return false;
}

} // end namespace llvm
