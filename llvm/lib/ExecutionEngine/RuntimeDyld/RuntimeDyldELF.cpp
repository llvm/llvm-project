//===-- RuntimeDyldELF.cpp - Run-time dynamic linker for MC-JIT -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of ELF support for the MC-JIT runtime dynamic linker.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "dyld"
#include "RuntimeDyldELF.h"
#include "JITRegistrar.h"
#include "ObjectImageCommon.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/ObjectBuffer.h"
#include "llvm/ExecutionEngine/ObjectImage.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/ELF.h"
using namespace llvm;
using namespace llvm::object;

namespace {

static inline
error_code check(error_code Err) {
  if (Err) {
    report_fatal_error(Err.message());
  }
  return Err;
}

template<class ELFT>
class DyldELFObject
  : public ELFObjectFile<ELFT> {
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)

  typedef Elf_Shdr_Impl<ELFT> Elf_Shdr;
  typedef Elf_Sym_Impl<ELFT> Elf_Sym;
  typedef
    Elf_Rel_Impl<ELFT, false> Elf_Rel;
  typedef
    Elf_Rel_Impl<ELFT, true> Elf_Rela;

  typedef Elf_Ehdr_Impl<ELFT> Elf_Ehdr;

  typedef typename ELFDataTypeTypedefHelper<
          ELFT>::value_type addr_type;

public:
  DyldELFObject(MemoryBuffer *Wrapper, error_code &ec);

  void updateSectionAddress(const SectionRef &Sec, uint64_t Addr);
  void updateSymbolAddress(const SymbolRef &Sym, uint64_t Addr);

  // Methods for type inquiry through isa, cast and dyn_cast
  static inline bool classof(const Binary *v) {
    return (isa<ELFObjectFile<ELFT> >(v)
            && classof(cast<ELFObjectFile
                <ELFT> >(v)));
  }
  static inline bool classof(
      const ELFObjectFile<ELFT> *v) {
    return v->isDyldType();
  }
};

template<class ELFT>
class ELFObjectImage : public ObjectImageCommon {
  protected:
    DyldELFObject<ELFT> *DyldObj;
    bool Registered;

  public:
    ELFObjectImage(ObjectBuffer *Input,
                 DyldELFObject<ELFT> *Obj)
    : ObjectImageCommon(Input, Obj),
      DyldObj(Obj),
      Registered(false) {}

    virtual ~ELFObjectImage() {
      if (Registered)
        deregisterWithDebugger();
    }

    // Subclasses can override these methods to update the image with loaded
    // addresses for sections and common symbols
    virtual void updateSectionAddress(const SectionRef &Sec, uint64_t Addr)
    {
      DyldObj->updateSectionAddress(Sec, Addr);
    }

    virtual void updateSymbolAddress(const SymbolRef &Sym, uint64_t Addr)
    {
      DyldObj->updateSymbolAddress(Sym, Addr);
    }

    virtual void registerWithDebugger()
    {
      JITRegistrar::getGDBRegistrar().registerObject(*Buffer);
      Registered = true;
    }
    virtual void deregisterWithDebugger()
    {
      JITRegistrar::getGDBRegistrar().deregisterObject(*Buffer);
    }
};

// The MemoryBuffer passed into this constructor is just a wrapper around the
// actual memory.  Ultimately, the Binary parent class will take ownership of
// this MemoryBuffer object but not the underlying memory.
template<class ELFT>
DyldELFObject<ELFT>::DyldELFObject(MemoryBuffer *Wrapper, error_code &ec)
  : ELFObjectFile<ELFT>(Wrapper, ec) {
  this->isDyldELFObject = true;
}

template<class ELFT>
void DyldELFObject<ELFT>::updateSectionAddress(const SectionRef &Sec,
                                               uint64_t Addr) {
  DataRefImpl ShdrRef = Sec.getRawDataRefImpl();
  Elf_Shdr *shdr = const_cast<Elf_Shdr*>(
                          reinterpret_cast<const Elf_Shdr *>(ShdrRef.p));

  // This assumes the address passed in matches the target address bitness
  // The template-based type cast handles everything else.
  shdr->sh_addr = static_cast<addr_type>(Addr);
}

template<class ELFT>
void DyldELFObject<ELFT>::updateSymbolAddress(const SymbolRef &SymRef,
                                              uint64_t Addr) {

  Elf_Sym *sym = const_cast<Elf_Sym*>(
    ELFObjectFile<ELFT>::getSymbol(SymRef.getRawDataRefImpl()));

  // This assumes the address passed in matches the target address bitness
  // The template-based type cast handles everything else.
  sym->st_value = static_cast<addr_type>(Addr);
}

} // namespace

namespace llvm {

void RuntimeDyldELF::registerEHFrames() {
  if (!MemMgr)
    return;
  for (int i = 0, e = UnregisteredEHFrameSections.size(); i != e; ++i) {
    SID EHFrameSID = UnregisteredEHFrameSections[i];
    uint8_t *EHFrameAddr = Sections[EHFrameSID].Address;
    uint64_t EHFrameLoadAddr = Sections[EHFrameSID].LoadAddress;
    size_t EHFrameSize = Sections[EHFrameSID].Size;
    MemMgr->registerEHFrames(EHFrameAddr, EHFrameLoadAddr, EHFrameSize);
    RegisteredEHFrameSections.push_back(EHFrameSID);
  }
  UnregisteredEHFrameSections.clear();
}

void RuntimeDyldELF::deregisterEHFrames() {
  if (!MemMgr)
    return;
  for (int i = 0, e = RegisteredEHFrameSections.size(); i != e; ++i) {
    SID EHFrameSID = RegisteredEHFrameSections[i];
    uint8_t *EHFrameAddr = Sections[EHFrameSID].Address;
    uint64_t EHFrameLoadAddr = Sections[EHFrameSID].LoadAddress;
    size_t EHFrameSize = Sections[EHFrameSID].Size;
    MemMgr->deregisterEHFrames(EHFrameAddr, EHFrameLoadAddr, EHFrameSize);
  }
  RegisteredEHFrameSections.clear();
}

ObjectImage *RuntimeDyldELF::createObjectImage(ObjectBuffer *Buffer) {
  if (Buffer->getBufferSize() < ELF::EI_NIDENT)
    llvm_unreachable("Unexpected ELF object size");
  std::pair<unsigned char, unsigned char> Ident = std::make_pair(
                         (uint8_t)Buffer->getBufferStart()[ELF::EI_CLASS],
                         (uint8_t)Buffer->getBufferStart()[ELF::EI_DATA]);
  error_code ec;

  if (Ident.first == ELF::ELFCLASS32 && Ident.second == ELF::ELFDATA2LSB) {
    DyldELFObject<ELFType<support::little, 4, false> > *Obj =
      new DyldELFObject<ELFType<support::little, 4, false> >(
        Buffer->getMemBuffer(), ec);
    return new ELFObjectImage<ELFType<support::little, 4, false> >(Buffer, Obj);
  }
  else if (Ident.first == ELF::ELFCLASS32 && Ident.second == ELF::ELFDATA2MSB) {
    DyldELFObject<ELFType<support::big, 4, false> > *Obj =
      new DyldELFObject<ELFType<support::big, 4, false> >(
        Buffer->getMemBuffer(), ec);
    return new ELFObjectImage<ELFType<support::big, 4, false> >(Buffer, Obj);
  }
  else if (Ident.first == ELF::ELFCLASS64 && Ident.second == ELF::ELFDATA2MSB) {
    DyldELFObject<ELFType<support::big, 8, true> > *Obj =
      new DyldELFObject<ELFType<support::big, 8, true> >(
        Buffer->getMemBuffer(), ec);
    return new ELFObjectImage<ELFType<support::big, 8, true> >(Buffer, Obj);
  }
  else if (Ident.first == ELF::ELFCLASS64 && Ident.second == ELF::ELFDATA2LSB) {
    DyldELFObject<ELFType<support::little, 8, true> > *Obj =
      new DyldELFObject<ELFType<support::little, 8, true> >(
        Buffer->getMemBuffer(), ec);
    return new ELFObjectImage<ELFType<support::little, 8, true> >(Buffer, Obj);
  }
  else
    llvm_unreachable("Unexpected ELF format");
}

RuntimeDyldELF::~RuntimeDyldELF() {
}

void RuntimeDyldELF::resolveX86_64Relocation(const SectionEntry &Section,
                                             uint64_t Offset,
                                             uint64_t Value,
                                             uint32_t Type,
                                             int64_t  Addend,
                                             uint64_t SymOffset) {
  switch (Type) {
  default:
    llvm_unreachable("Relocation type not implemented yet!");
  break;
  case ELF::R_X86_64_64: {
    uint64_t *Target = reinterpret_cast<uint64_t*>(Section.Address + Offset);
    *Target = Value + Addend;
    DEBUG(dbgs() << "Writing " << format("%p", (Value + Addend))
                 << " at " << format("%p\n",Target));
    break;
  }
  case ELF::R_X86_64_32:
  case ELF::R_X86_64_32S: {
    Value += Addend;
    assert((Type == ELF::R_X86_64_32 && (Value <= UINT32_MAX)) ||
           (Type == ELF::R_X86_64_32S &&
             ((int64_t)Value <= INT32_MAX && (int64_t)Value >= INT32_MIN)));
    uint32_t TruncatedAddr = (Value & 0xFFFFFFFF);
    uint32_t *Target = reinterpret_cast<uint32_t*>(Section.Address + Offset);
    *Target = TruncatedAddr;
    DEBUG(dbgs() << "Writing " << format("%p", TruncatedAddr)
                 << " at " << format("%p\n",Target));
    break;
  }
  case ELF::R_X86_64_GOTPCREL: {
    // findGOTEntry returns the 'G + GOT' part of the relocation calculation
    // based on the load/target address of the GOT (not the current/local addr).
    uint64_t GOTAddr = findGOTEntry(Value, SymOffset);
    uint32_t *Target = reinterpret_cast<uint32_t*>(Section.Address + Offset);
    uint64_t  FinalAddress = Section.LoadAddress + Offset;
    // The processRelocationRef method combines the symbol offset and the addend
    // and in most cases that's what we want.  For this relocation type, we need
    // the raw addend, so we subtract the symbol offset to get it.
    int64_t RealOffset = GOTAddr + Addend - SymOffset - FinalAddress;
    assert(RealOffset <= INT32_MAX && RealOffset >= INT32_MIN);
    int32_t TruncOffset = (RealOffset & 0xFFFFFFFF);
    *Target = TruncOffset;
    break;
  }
  case ELF::R_X86_64_PC32: {
    // Get the placeholder value from the generated object since
    // a previous relocation attempt may have overwritten the loaded version
    uint32_t *Placeholder = reinterpret_cast<uint32_t*>(Section.ObjAddress
                                                                   + Offset);
    uint32_t *Target = reinterpret_cast<uint32_t*>(Section.Address + Offset);
    uint64_t  FinalAddress = Section.LoadAddress + Offset;
    int64_t RealOffset = *Placeholder + Value + Addend - FinalAddress;
    assert(RealOffset <= INT32_MAX && RealOffset >= INT32_MIN);
    int32_t TruncOffset = (RealOffset & 0xFFFFFFFF);
    *Target = TruncOffset;
    break;
  }
  case ELF::R_X86_64_PC64: {
    // Get the placeholder value from the generated object since
    // a previous relocation attempt may have overwritten the loaded version
    uint64_t *Placeholder = reinterpret_cast<uint64_t*>(Section.ObjAddress
                                                                   + Offset);
    uint64_t *Target = reinterpret_cast<uint64_t*>(Section.Address + Offset);
    uint64_t  FinalAddress = Section.LoadAddress + Offset;
    *Target = *Placeholder + Value + Addend - FinalAddress;
    break;
  }
  }
}

void RuntimeDyldELF::resolveX86Relocation(const SectionEntry &Section,
                                          uint64_t Offset,
                                          uint32_t Value,
                                          uint32_t Type,
                                          int32_t Addend) {
  switch (Type) {
  case ELF::R_386_32: {
    // Get the placeholder value from the generated object since
    // a previous relocation attempt may have overwritten the loaded version
    uint32_t *Placeholder = reinterpret_cast<uint32_t*>(Section.ObjAddress
                                                                   + Offset);
    uint32_t *Target = reinterpret_cast<uint32_t*>(Section.Address + Offset);
    *Target = *Placeholder + Value + Addend;
    break;
  }
  case ELF::R_386_PC32: {
    // Get the placeholder value from the generated object since
    // a previous relocation attempt may have overwritten the loaded version
    uint32_t *Placeholder = reinterpret_cast<uint32_t*>(Section.ObjAddress
                                                                   + Offset);
    uint32_t *Target = reinterpret_cast<uint32_t*>(Section.Address + Offset);
    uint32_t  FinalAddress = ((Section.LoadAddress + Offset) & 0xFFFFFFFF);
    uint32_t RealOffset = *Placeholder + Value + Addend - FinalAddress;
    *Target = RealOffset;
    break;
    }
    default:
      // There are other relocation types, but it appears these are the
      // only ones currently used by the LLVM ELF object writer
      llvm_unreachable("Relocation type not implemented yet!");
      break;
  }
}

void RuntimeDyldELF::resolveAArch64Relocation(const SectionEntry &Section,
                                              uint64_t Offset,
                                              uint64_t Value,
                                              uint32_t Type,
                                              int64_t Addend) {
  uint32_t *TargetPtr = reinterpret_cast<uint32_t*>(Section.Address + Offset);
  uint64_t FinalAddress = Section.LoadAddress + Offset;

  DEBUG(dbgs() << "resolveAArch64Relocation, LocalAddress: 0x"
               << format("%llx", Section.Address + Offset)
               << " FinalAddress: 0x" << format("%llx",FinalAddress)
               << " Value: 0x" << format("%llx",Value)
               << " Type: 0x" << format("%x",Type)
               << " Addend: 0x" << format("%llx",Addend)
               << "\n");

  switch (Type) {
  default:
    llvm_unreachable("Relocation type not implemented yet!");
    break;
  case ELF::R_AARCH64_ABS64: {
    uint64_t *TargetPtr = reinterpret_cast<uint64_t*>(Section.Address + Offset);
    *TargetPtr = Value + Addend;
    break;
  }
  case ELF::R_AARCH64_PREL32: {
    uint64_t Result = Value + Addend - FinalAddress;
    assert(static_cast<int64_t>(Result) >= INT32_MIN &&
           static_cast<int64_t>(Result) <= UINT32_MAX);
    *TargetPtr = static_cast<uint32_t>(Result & 0xffffffffU);
    break;
  }
  case ELF::R_AARCH64_CALL26: // fallthrough
  case ELF::R_AARCH64_JUMP26: {
    // Operation: S+A-P. Set Call or B immediate value to bits fff_fffc of the
    // calculation.
    uint64_t BranchImm = Value + Addend - FinalAddress;

    // "Check that -2^27 <= result < 2^27".
    assert(-(1LL << 27) <= static_cast<int64_t>(BranchImm) &&
           static_cast<int64_t>(BranchImm) < (1LL << 27));

    // AArch64 code is emitted with .rela relocations. The data already in any
    // bits affected by the relocation on entry is garbage.
    *TargetPtr &= 0xfc000000U;
    // Immediate goes in bits 25:0 of B and BL.
    *TargetPtr |= static_cast<uint32_t>(BranchImm & 0xffffffcU) >> 2;
    break;
  }
  case ELF::R_AARCH64_MOVW_UABS_G3: {
    uint64_t Result = Value + Addend;

    // AArch64 code is emitted with .rela relocations. The data already in any
    // bits affected by the relocation on entry is garbage.
    *TargetPtr &= 0xffe0001fU;
    // Immediate goes in bits 20:5 of MOVZ/MOVK instruction
    *TargetPtr |= Result >> (48 - 5);
    // Shift must be "lsl #48", in bits 22:21
    assert((*TargetPtr >> 21 & 0x3) == 3 && "invalid shift for relocation");
    break;
  }
  case ELF::R_AARCH64_MOVW_UABS_G2_NC: {
    uint64_t Result = Value + Addend;

    // AArch64 code is emitted with .rela relocations. The data already in any
    // bits affected by the relocation on entry is garbage.
    *TargetPtr &= 0xffe0001fU;
    // Immediate goes in bits 20:5 of MOVZ/MOVK instruction
    *TargetPtr |= ((Result & 0xffff00000000ULL) >> (32 - 5));
    // Shift must be "lsl #32", in bits 22:21
    assert((*TargetPtr >> 21 & 0x3) == 2 && "invalid shift for relocation");
    break;
  }
  case ELF::R_AARCH64_MOVW_UABS_G1_NC: {
    uint64_t Result = Value + Addend;

    // AArch64 code is emitted with .rela relocations. The data already in any
    // bits affected by the relocation on entry is garbage.
    *TargetPtr &= 0xffe0001fU;
    // Immediate goes in bits 20:5 of MOVZ/MOVK instruction
    *TargetPtr |= ((Result & 0xffff0000U) >> (16 - 5));
    // Shift must be "lsl #16", in bits 22:2
    assert((*TargetPtr >> 21 & 0x3) == 1 && "invalid shift for relocation");
    break;
  }
  case ELF::R_AARCH64_MOVW_UABS_G0_NC: {
    uint64_t Result = Value + Addend;

    // AArch64 code is emitted with .rela relocations. The data already in any
    // bits affected by the relocation on entry is garbage.
    *TargetPtr &= 0xffe0001fU;
    // Immediate goes in bits 20:5 of MOVZ/MOVK instruction
    *TargetPtr |= ((Result & 0xffffU) << 5);
    // Shift must be "lsl #0", in bits 22:21.
    assert((*TargetPtr >> 21 & 0x3) == 0 && "invalid shift for relocation");
    break;
  }
  }
}

void RuntimeDyldELF::resolveARMRelocation(const SectionEntry &Section,
                                          uint64_t Offset,
                                          uint32_t Value,
                                          uint32_t Type,
                                          int32_t Addend) {
  // TODO: Add Thumb relocations.
  uint32_t *Placeholder = reinterpret_cast<uint32_t*>(Section.ObjAddress +
                                                      Offset);
  uint32_t* TargetPtr = (uint32_t*)(Section.Address + Offset);
  uint32_t FinalAddress = ((Section.LoadAddress + Offset) & 0xFFFFFFFF);
  Value += Addend;

  DEBUG(dbgs() << "resolveARMRelocation, LocalAddress: "
               << Section.Address + Offset
               << " FinalAddress: " << format("%p",FinalAddress)
               << " Value: " << format("%x",Value)
               << " Type: " << format("%x",Type)
               << " Addend: " << format("%x",Addend)
               << "\n");

  switch(Type) {
  default:
    llvm_unreachable("Not implemented relocation type!");

  // Write a 32bit value to relocation address, taking into account the
  // implicit addend encoded in the target.
  case ELF::R_ARM_TARGET1:
  case ELF::R_ARM_ABS32:
    *TargetPtr = *Placeholder + Value;
    break;
  // Write first 16 bit of 32 bit value to the mov instruction.
  // Last 4 bit should be shifted.
  case ELF::R_ARM_MOVW_ABS_NC:
    // We are not expecting any other addend in the relocation address.
    // Using 0x000F0FFF because MOVW has its 16 bit immediate split into 2
    // non-contiguous fields.
    assert((*Placeholder & 0x000F0FFF) == 0);
    Value = Value & 0xFFFF;
    *TargetPtr = *Placeholder | (Value & 0xFFF);
    *TargetPtr |= ((Value >> 12) & 0xF) << 16;
    break;
  // Write last 16 bit of 32 bit value to the mov instruction.
  // Last 4 bit should be shifted.
  case ELF::R_ARM_MOVT_ABS:
    // We are not expecting any other addend in the relocation address.
    // Use 0x000F0FFF for the same reason as R_ARM_MOVW_ABS_NC.
    assert((*Placeholder & 0x000F0FFF) == 0);

    Value = (Value >> 16) & 0xFFFF;
    *TargetPtr = *Placeholder | (Value & 0xFFF);
    *TargetPtr |= ((Value >> 12) & 0xF) << 16;
    break;
  // Write 24 bit relative value to the branch instruction.
  case ELF::R_ARM_PC24 :    // Fall through.
  case ELF::R_ARM_CALL :    // Fall through.
  case ELF::R_ARM_JUMP24: {
    int32_t RelValue = static_cast<int32_t>(Value - FinalAddress - 8);
    RelValue = (RelValue & 0x03FFFFFC) >> 2;
    assert((*TargetPtr & 0xFFFFFF) == 0xFFFFFE);
    *TargetPtr &= 0xFF000000;
    *TargetPtr |= RelValue;
    break;
  }
  case ELF::R_ARM_PRIVATE_0:
    // This relocation is reserved by the ARM ELF ABI for internal use. We
    // appropriate it here to act as an R_ARM_ABS32 without any addend for use
    // in the stubs created during JIT (which can't put an addend into the
    // original object file).
    *TargetPtr = Value;
    break;
  }
}

void RuntimeDyldELF::resolveMIPSRelocation(const SectionEntry &Section,
                                           uint64_t Offset,
                                           uint32_t Value,
                                           uint32_t Type,
                                           int32_t Addend) {
  uint32_t *Placeholder = reinterpret_cast<uint32_t*>(Section.ObjAddress +
                                                      Offset);
  uint32_t* TargetPtr = (uint32_t*)(Section.Address + Offset);
  Value += Addend;

  DEBUG(dbgs() << "resolveMipselocation, LocalAddress: "
               << Section.Address + Offset
               << " FinalAddress: "
               << format("%p",Section.LoadAddress + Offset)
               << " Value: " << format("%x",Value)
               << " Type: " << format("%x",Type)
               << " Addend: " << format("%x",Addend)
               << "\n");

  switch(Type) {
  default:
    llvm_unreachable("Not implemented relocation type!");
    break;
  case ELF::R_MIPS_32:
    *TargetPtr = Value + (*Placeholder);
    break;
  case ELF::R_MIPS_26:
    *TargetPtr = ((*Placeholder) & 0xfc000000) | (( Value & 0x0fffffff) >> 2);
    break;
  case ELF::R_MIPS_HI16:
    // Get the higher 16-bits. Also add 1 if bit 15 is 1.
    Value += ((*Placeholder) & 0x0000ffff) << 16;
    *TargetPtr = ((*Placeholder) & 0xffff0000) |
                 (((Value + 0x8000) >> 16) & 0xffff);
    break;
  case ELF::R_MIPS_LO16:
    Value += ((*Placeholder) & 0x0000ffff);
    *TargetPtr = ((*Placeholder) & 0xffff0000) | (Value & 0xffff);
    break;
  case ELF::R_MIPS_UNUSED1:
    // Similar to ELF::R_ARM_PRIVATE_0, R_MIPS_UNUSED1 and R_MIPS_UNUSED2
    // are used for internal JIT purpose. These relocations are similar to
    // R_MIPS_HI16 and R_MIPS_LO16, but they do not take any addend into
    // account.
    *TargetPtr = ((*TargetPtr) & 0xffff0000) |
                 (((Value + 0x8000) >> 16) & 0xffff);
    break;
  case ELF::R_MIPS_UNUSED2:
    *TargetPtr = ((*TargetPtr) & 0xffff0000) | (Value & 0xffff);
    break;
   }
}

// Return the .TOC. section address to R_PPC64_TOC relocations.
uint64_t RuntimeDyldELF::findPPC64TOC() const {
  // The TOC consists of sections .got, .toc, .tocbss, .plt in that
  // order. The TOC starts where the first of these sections starts.
  SectionList::const_iterator it = Sections.begin();
  SectionList::const_iterator ite = Sections.end();
  for (; it != ite; ++it) {
    if (it->Name == ".got" ||
        it->Name == ".toc" ||
        it->Name == ".tocbss" ||
        it->Name == ".plt")
      break;
  }
  if (it == ite) {
    // This may happen for
    // * references to TOC base base (sym@toc, .odp relocation) without
    // a .toc directive.
    // In this case just use the first section (which is usually
    // the .odp) since the code won't reference the .toc base
    // directly.
    it = Sections.begin();
  }
  assert (it != ite);
  // Per the ppc64-elf-linux ABI, The TOC base is TOC value plus 0x8000
  // thus permitting a full 64 Kbytes segment.
  return it->LoadAddress + 0x8000;
}

// Returns the sections and offset associated with the ODP entry referenced
// by Symbol.
void RuntimeDyldELF::findOPDEntrySection(ObjectImage &Obj,
                                         ObjSectionToIDMap &LocalSections,
                                         RelocationValueRef &Rel) {
  // Get the ELF symbol value (st_value) to compare with Relocation offset in
  // .opd entries

  error_code err;
  for (section_iterator si = Obj.begin_sections(),
     se = Obj.end_sections(); si != se; si.increment(err)) {
    section_iterator RelSecI = si->getRelocatedSection();
    if (RelSecI == Obj.end_sections())
      continue;

    StringRef RelSectionName;
    check(RelSecI->getName(RelSectionName));
    if (RelSectionName != ".opd")
      continue;

    for (relocation_iterator i = si->begin_relocations(),
         e = si->end_relocations(); i != e;) {
      check(err);

      // The R_PPC64_ADDR64 relocation indicates the first field
      // of a .opd entry
      uint64_t TypeFunc;
      check(i->getType(TypeFunc));
      if (TypeFunc != ELF::R_PPC64_ADDR64) {
        i.increment(err);
        continue;
      }

      uint64_t TargetSymbolOffset;
      symbol_iterator TargetSymbol = i->getSymbol();
      check(i->getOffset(TargetSymbolOffset));
      int64_t Addend;
      check(getELFRelocationAddend(*i, Addend));

      i = i.increment(err);
      if (i == e)
        break;
      check(err);

      // Just check if following relocation is a R_PPC64_TOC
      uint64_t TypeTOC;
      check(i->getType(TypeTOC));
      if (TypeTOC != ELF::R_PPC64_TOC)
        continue;

      // Finally compares the Symbol value and the target symbol offset
      // to check if this .opd entry refers to the symbol the relocation
      // points to.
      if (Rel.Addend != (int64_t)TargetSymbolOffset)
        continue;

      section_iterator tsi(Obj.end_sections());
      check(TargetSymbol->getSection(tsi));
      Rel.SectionID = findOrEmitSection(Obj, (*tsi), true, LocalSections);
      Rel.Addend = (intptr_t)Addend;
      return;
    }
  }
  llvm_unreachable("Attempting to get address of ODP entry!");
}

// Relocation masks following the #lo(value), #hi(value), #higher(value),
// and #highest(value) macros defined in section 4.5.1. Relocation Types
// in PPC-elf64abi document.
//
static inline
uint16_t applyPPClo (uint64_t value)
{
  return value & 0xffff;
}

static inline
uint16_t applyPPChi (uint64_t value)
{
  return (value >> 16) & 0xffff;
}

static inline
uint16_t applyPPChigher (uint64_t value)
{
  return (value >> 32) & 0xffff;
}

static inline
uint16_t applyPPChighest (uint64_t value)
{
  return (value >> 48) & 0xffff;
}

void RuntimeDyldELF::resolvePPC64Relocation(const SectionEntry &Section,
                                            uint64_t Offset,
                                            uint64_t Value,
                                            uint32_t Type,
                                            int64_t Addend) {
  uint8_t* LocalAddress = Section.Address + Offset;
  switch (Type) {
  default:
    llvm_unreachable("Relocation type not implemented yet!");
  break;
  case ELF::R_PPC64_ADDR16_LO :
    writeInt16BE(LocalAddress, applyPPClo (Value + Addend));
    break;
  case ELF::R_PPC64_ADDR16_HI :
    writeInt16BE(LocalAddress, applyPPChi (Value + Addend));
    break;
  case ELF::R_PPC64_ADDR16_HIGHER :
    writeInt16BE(LocalAddress, applyPPChigher (Value + Addend));
    break;
  case ELF::R_PPC64_ADDR16_HIGHEST :
    writeInt16BE(LocalAddress, applyPPChighest (Value + Addend));
    break;
  case ELF::R_PPC64_ADDR14 : {
    assert(((Value + Addend) & 3) == 0);
    // Preserve the AA/LK bits in the branch instruction
    uint8_t aalk = *(LocalAddress+3);
    writeInt16BE(LocalAddress + 2, (aalk & 3) | ((Value + Addend) & 0xfffc));
  } break;
  case ELF::R_PPC64_ADDR32 : {
    int32_t Result = static_cast<int32_t>(Value + Addend);
    if (SignExtend32<32>(Result) != Result)
      llvm_unreachable("Relocation R_PPC64_ADDR32 overflow");
    writeInt32BE(LocalAddress, Result);
  } break;
  case ELF::R_PPC64_REL24 : {
    uint64_t FinalAddress = (Section.LoadAddress + Offset);
    int32_t delta = static_cast<int32_t>(Value - FinalAddress + Addend);
    if (SignExtend32<24>(delta) != delta)
      llvm_unreachable("Relocation R_PPC64_REL24 overflow");
    // Generates a 'bl <address>' instruction
    writeInt32BE(LocalAddress, 0x48000001 | (delta & 0x03FFFFFC));
  } break;
  case ELF::R_PPC64_REL32 : {
    uint64_t FinalAddress = (Section.LoadAddress + Offset);
    int32_t delta = static_cast<int32_t>(Value - FinalAddress + Addend);
    if (SignExtend32<32>(delta) != delta)
      llvm_unreachable("Relocation R_PPC64_REL32 overflow");
    writeInt32BE(LocalAddress, delta);
  } break;
  case ELF::R_PPC64_REL64: {
    uint64_t FinalAddress = (Section.LoadAddress + Offset);
    uint64_t Delta = Value - FinalAddress + Addend;
    writeInt64BE(LocalAddress, Delta);
  } break;
  case ELF::R_PPC64_ADDR64 :
    writeInt64BE(LocalAddress, Value + Addend);
    break;
  case ELF::R_PPC64_TOC :
    writeInt64BE(LocalAddress, findPPC64TOC());
    break;
  case ELF::R_PPC64_TOC16 : {
    uint64_t TOCStart = findPPC64TOC();
    Value = applyPPClo((Value + Addend) - TOCStart);
    writeInt16BE(LocalAddress, applyPPClo(Value));
  } break;
  case ELF::R_PPC64_TOC16_DS : {
    uint64_t TOCStart = findPPC64TOC();
    Value = ((Value + Addend) - TOCStart);
    writeInt16BE(LocalAddress, applyPPClo(Value));
  } break;
  }
}

void RuntimeDyldELF::resolveSystemZRelocation(const SectionEntry &Section,
                                              uint64_t Offset,
                                              uint64_t Value,
                                              uint32_t Type,
                                              int64_t Addend) {
  uint8_t *LocalAddress = Section.Address + Offset;
  switch (Type) {
  default:
    llvm_unreachable("Relocation type not implemented yet!");
    break;
  case ELF::R_390_PC16DBL:
  case ELF::R_390_PLT16DBL: {
    int64_t Delta = (Value + Addend) - (Section.LoadAddress + Offset);
    assert(int16_t(Delta / 2) * 2 == Delta && "R_390_PC16DBL overflow");
    writeInt16BE(LocalAddress, Delta / 2);
    break;
  }
  case ELF::R_390_PC32DBL:
  case ELF::R_390_PLT32DBL: {
    int64_t Delta = (Value + Addend) - (Section.LoadAddress + Offset);
    assert(int32_t(Delta / 2) * 2 == Delta && "R_390_PC32DBL overflow");
    writeInt32BE(LocalAddress, Delta / 2);
    break;
  }
  case ELF::R_390_PC32: {
    int64_t Delta = (Value + Addend) - (Section.LoadAddress + Offset);
    assert(int32_t(Delta) == Delta && "R_390_PC32 overflow");
    writeInt32BE(LocalAddress, Delta);
    break;
  }
  case ELF::R_390_64:
    writeInt64BE(LocalAddress, Value + Addend);
    break;
  }
}

// The target location for the relocation is described by RE.SectionID and
// RE.Offset.  RE.SectionID can be used to find the SectionEntry.  Each
// SectionEntry has three members describing its location.
// SectionEntry::Address is the address at which the section has been loaded
// into memory in the current (host) process.  SectionEntry::LoadAddress is the
// address that the section will have in the target process.
// SectionEntry::ObjAddress is the address of the bits for this section in the
// original emitted object image (also in the current address space).
//
// Relocations will be applied as if the section were loaded at
// SectionEntry::LoadAddress, but they will be applied at an address based
// on SectionEntry::Address.  SectionEntry::ObjAddress will be used to refer to
// Target memory contents if they are required for value calculations.
//
// The Value parameter here is the load address of the symbol for the
// relocation to be applied.  For relocations which refer to symbols in the
// current object Value will be the LoadAddress of the section in which
// the symbol resides (RE.Addend provides additional information about the
// symbol location).  For external symbols, Value will be the address of the
// symbol in the target address space.
void RuntimeDyldELF::resolveRelocation(const RelocationEntry &RE,
                                       uint64_t Value) {
  const SectionEntry &Section = Sections[RE.SectionID];
  return resolveRelocation(Section, RE.Offset, Value, RE.RelType, RE.Addend,
                           RE.SymOffset);
}

void RuntimeDyldELF::resolveRelocation(const SectionEntry &Section,
                                       uint64_t Offset,
                                       uint64_t Value,
                                       uint32_t Type,
                                       int64_t  Addend,
                                       uint64_t SymOffset) {
  switch (Arch) {
  case Triple::x86_64:
    resolveX86_64Relocation(Section, Offset, Value, Type, Addend, SymOffset);
    break;
  case Triple::x86:
    resolveX86Relocation(Section, Offset,
                         (uint32_t)(Value & 0xffffffffL), Type,
                         (uint32_t)(Addend & 0xffffffffL));
    break;
  case Triple::aarch64:
    resolveAArch64Relocation(Section, Offset, Value, Type, Addend);
    break;
  case Triple::arm:    // Fall through.
  case Triple::thumb:
    resolveARMRelocation(Section, Offset,
                         (uint32_t)(Value & 0xffffffffL), Type,
                         (uint32_t)(Addend & 0xffffffffL));
    break;
  case Triple::mips:    // Fall through.
  case Triple::mipsel:
    resolveMIPSRelocation(Section, Offset,
                          (uint32_t)(Value & 0xffffffffL), Type,
                          (uint32_t)(Addend & 0xffffffffL));
    break;
  case Triple::ppc64:   // Fall through.
  case Triple::ppc64le:
    resolvePPC64Relocation(Section, Offset, Value, Type, Addend);
    break;
  case Triple::systemz:
    resolveSystemZRelocation(Section, Offset, Value, Type, Addend);
    break;
  default: llvm_unreachable("Unsupported CPU type!");
  }
}

void RuntimeDyldELF::processRelocationRef(unsigned SectionID,
                                          RelocationRef RelI,
                                          ObjectImage &Obj,
                                          ObjSectionToIDMap &ObjSectionToID,
                                          const SymbolTableMap &Symbols,
                                          StubMap &Stubs) {
  uint64_t RelType;
  Check(RelI.getType(RelType));
  int64_t Addend;
  Check(getELFRelocationAddend(RelI, Addend));
  symbol_iterator Symbol = RelI.getSymbol();

  // Obtain the symbol name which is referenced in the relocation
  StringRef TargetName;
  if (Symbol != Obj.end_symbols())
    Symbol->getName(TargetName);
  DEBUG(dbgs() << "\t\tRelType: " << RelType
               << " Addend: " << Addend
               << " TargetName: " << TargetName
               << "\n");
  RelocationValueRef Value;
  // First search for the symbol in the local symbol table
  SymbolTableMap::const_iterator lsi = Symbols.end();
  SymbolRef::Type SymType = SymbolRef::ST_Unknown;
  if (Symbol != Obj.end_symbols()) {
    lsi = Symbols.find(TargetName.data());
    Symbol->getType(SymType);
  }
  if (lsi != Symbols.end()) {
    Value.SectionID = lsi->second.first;
    Value.Offset = lsi->second.second;
    Value.Addend = lsi->second.second + Addend;
  } else {
    // Search for the symbol in the global symbol table
    SymbolTableMap::const_iterator gsi = GlobalSymbolTable.end();
    if (Symbol != Obj.end_symbols())
      gsi = GlobalSymbolTable.find(TargetName.data());
    if (gsi != GlobalSymbolTable.end()) {
      Value.SectionID = gsi->second.first;
      Value.Offset = gsi->second.second;
      Value.Addend = gsi->second.second + Addend;
    } else {
      switch (SymType) {
        case SymbolRef::ST_Debug: {
          // TODO: Now ELF SymbolRef::ST_Debug = STT_SECTION, it's not obviously
          // and can be changed by another developers. Maybe best way is add
          // a new symbol type ST_Section to SymbolRef and use it.
          section_iterator si(Obj.end_sections());
          Symbol->getSection(si);
          if (si == Obj.end_sections())
            llvm_unreachable("Symbol section not found, bad object file format!");
          DEBUG(dbgs() << "\t\tThis is section symbol\n");
          // Default to 'true' in case isText fails (though it never does).
          bool isCode = true;
          si->isText(isCode);
          Value.SectionID = findOrEmitSection(Obj,
                                              (*si),
                                              isCode,
                                              ObjSectionToID);
          Value.Addend = Addend;
          break;
        }
        case SymbolRef::ST_Data:
        case SymbolRef::ST_Unknown: {
          Value.SymbolName = TargetName.data();
          Value.Addend = Addend;

          // Absolute relocations will have a zero symbol ID (STN_UNDEF), which
          // will manifest here as a NULL symbol name.
          // We can set this as a valid (but empty) symbol name, and rely
          // on addRelocationForSymbol to handle this.
          if (!Value.SymbolName)
              Value.SymbolName = "";
          break;
        }
        default:
          llvm_unreachable("Unresolved symbol type!");
          break;
      }
    }
  }
  uint64_t Offset;
  Check(RelI.getOffset(Offset));

  DEBUG(dbgs() << "\t\tSectionID: " << SectionID
               << " Offset: " << Offset
               << "\n");
  if (Arch == Triple::aarch64 &&
      (RelType == ELF::R_AARCH64_CALL26 ||
       RelType == ELF::R_AARCH64_JUMP26)) {
    // This is an AArch64 branch relocation, need to use a stub function.
    DEBUG(dbgs() << "\t\tThis is an AArch64 branch relocation.");
    SectionEntry &Section = Sections[SectionID];

    // Look for an existing stub.
    StubMap::const_iterator i = Stubs.find(Value);
    if (i != Stubs.end()) {
        resolveRelocation(Section, Offset,
                          (uint64_t)Section.Address + i->second, RelType, 0);
      DEBUG(dbgs() << " Stub function found\n");
    } else {
      // Create a new stub function.
      DEBUG(dbgs() << " Create a new stub function\n");
      Stubs[Value] = Section.StubOffset;
      uint8_t *StubTargetAddr = createStubFunction(Section.Address +
                                                   Section.StubOffset);

      RelocationEntry REmovz_g3(SectionID,
                                StubTargetAddr - Section.Address,
                                ELF::R_AARCH64_MOVW_UABS_G3, Value.Addend);
      RelocationEntry REmovk_g2(SectionID,
                                StubTargetAddr - Section.Address + 4,
                                ELF::R_AARCH64_MOVW_UABS_G2_NC, Value.Addend);
      RelocationEntry REmovk_g1(SectionID,
                                StubTargetAddr - Section.Address + 8,
                                ELF::R_AARCH64_MOVW_UABS_G1_NC, Value.Addend);
      RelocationEntry REmovk_g0(SectionID,
                                StubTargetAddr - Section.Address + 12,
                                ELF::R_AARCH64_MOVW_UABS_G0_NC, Value.Addend);

      if (Value.SymbolName) {
        addRelocationForSymbol(REmovz_g3, Value.SymbolName);
        addRelocationForSymbol(REmovk_g2, Value.SymbolName);
        addRelocationForSymbol(REmovk_g1, Value.SymbolName);
        addRelocationForSymbol(REmovk_g0, Value.SymbolName);
      } else {
        addRelocationForSection(REmovz_g3, Value.SectionID);
        addRelocationForSection(REmovk_g2, Value.SectionID);
        addRelocationForSection(REmovk_g1, Value.SectionID);
        addRelocationForSection(REmovk_g0, Value.SectionID);
      }
      resolveRelocation(Section, Offset,
                        (uint64_t)Section.Address + Section.StubOffset,
                        RelType, 0);
      Section.StubOffset += getMaxStubSize();
    }
  } else if (Arch == Triple::arm &&
      (RelType == ELF::R_ARM_PC24 ||
       RelType == ELF::R_ARM_CALL ||
       RelType == ELF::R_ARM_JUMP24)) {
    // This is an ARM branch relocation, need to use a stub function.
    DEBUG(dbgs() << "\t\tThis is an ARM branch relocation.");
    SectionEntry &Section = Sections[SectionID];

    // Look for an existing stub.
    StubMap::const_iterator i = Stubs.find(Value);
    if (i != Stubs.end()) {
        resolveRelocation(Section, Offset,
                          (uint64_t)Section.Address + i->second, RelType, 0);
      DEBUG(dbgs() << " Stub function found\n");
    } else {
      // Create a new stub function.
      DEBUG(dbgs() << " Create a new stub function\n");
      Stubs[Value] = Section.StubOffset;
      uint8_t *StubTargetAddr = createStubFunction(Section.Address +
                                                   Section.StubOffset);
      RelocationEntry RE(SectionID, StubTargetAddr - Section.Address,
                         ELF::R_ARM_PRIVATE_0, Value.Addend);
      if (Value.SymbolName)
        addRelocationForSymbol(RE, Value.SymbolName);
      else
        addRelocationForSection(RE, Value.SectionID);

      resolveRelocation(Section, Offset,
                        (uint64_t)Section.Address + Section.StubOffset,
                        RelType, 0);
      Section.StubOffset += getMaxStubSize();
    }
  } else if ((Arch == Triple::mipsel || Arch == Triple::mips) &&
             RelType == ELF::R_MIPS_26) {
    // This is an Mips branch relocation, need to use a stub function.
    DEBUG(dbgs() << "\t\tThis is a Mips branch relocation.");
    SectionEntry &Section = Sections[SectionID];
    uint8_t *Target = Section.Address + Offset;
    uint32_t *TargetAddress = (uint32_t *)Target;

    // Extract the addend from the instruction.
    uint32_t Addend = ((*TargetAddress) & 0x03ffffff) << 2;

    Value.Addend += Addend;

    //  Look up for existing stub.
    StubMap::const_iterator i = Stubs.find(Value);
    if (i != Stubs.end()) {
      RelocationEntry RE(SectionID, Offset, RelType, i->second);
      addRelocationForSection(RE, SectionID);
      DEBUG(dbgs() << " Stub function found\n");
    } else {
      // Create a new stub function.
      DEBUG(dbgs() << " Create a new stub function\n");
      Stubs[Value] = Section.StubOffset;
      uint8_t *StubTargetAddr = createStubFunction(Section.Address +
                                                   Section.StubOffset);

      // Creating Hi and Lo relocations for the filled stub instructions.
      RelocationEntry REHi(SectionID,
                           StubTargetAddr - Section.Address,
                           ELF::R_MIPS_UNUSED1, Value.Addend);
      RelocationEntry RELo(SectionID,
                           StubTargetAddr - Section.Address + 4,
                           ELF::R_MIPS_UNUSED2, Value.Addend);

      if (Value.SymbolName) {
        addRelocationForSymbol(REHi, Value.SymbolName);
        addRelocationForSymbol(RELo, Value.SymbolName);
      } else {
        addRelocationForSection(REHi, Value.SectionID);
        addRelocationForSection(RELo, Value.SectionID);
      }

      RelocationEntry RE(SectionID, Offset, RelType, Section.StubOffset);
      addRelocationForSection(RE, SectionID);
      Section.StubOffset += getMaxStubSize();
    }
  } else if (Arch == Triple::ppc64 || Arch == Triple::ppc64le) {
    if (RelType == ELF::R_PPC64_REL24) {
      // A PPC branch relocation will need a stub function if the target is
      // an external symbol (Symbol::ST_Unknown) or if the target address
      // is not within the signed 24-bits branch address.
      SectionEntry &Section = Sections[SectionID];
      uint8_t *Target = Section.Address + Offset;
      bool RangeOverflow = false;
      if (SymType != SymbolRef::ST_Unknown) {
        // A function call may points to the .opd entry, so the final symbol value
        // in calculated based in the relocation values in .opd section.
        findOPDEntrySection(Obj, ObjSectionToID, Value);
        uint8_t *RelocTarget = Sections[Value.SectionID].Address + Value.Addend;
        int32_t delta = static_cast<int32_t>(Target - RelocTarget);
        // If it is within 24-bits branch range, just set the branch target
        if (SignExtend32<24>(delta) == delta) {
          RelocationEntry RE(SectionID, Offset, RelType, Value.Addend);
          if (Value.SymbolName)
            addRelocationForSymbol(RE, Value.SymbolName);
          else
            addRelocationForSection(RE, Value.SectionID);
        } else {
          RangeOverflow = true;
        }
      }
      if (SymType == SymbolRef::ST_Unknown || RangeOverflow == true) {
        // It is an external symbol (SymbolRef::ST_Unknown) or within a range
        // larger than 24-bits.
        StubMap::const_iterator i = Stubs.find(Value);
        if (i != Stubs.end()) {
          // Symbol function stub already created, just relocate to it
          resolveRelocation(Section, Offset,
                            (uint64_t)Section.Address + i->second, RelType, 0);
          DEBUG(dbgs() << " Stub function found\n");
        } else {
          // Create a new stub function.
          DEBUG(dbgs() << " Create a new stub function\n");
          Stubs[Value] = Section.StubOffset;
          uint8_t *StubTargetAddr = createStubFunction(Section.Address +
                                                       Section.StubOffset);
          RelocationEntry RE(SectionID, StubTargetAddr - Section.Address,
                             ELF::R_PPC64_ADDR64, Value.Addend);

          // Generates the 64-bits address loads as exemplified in section
          // 4.5.1 in PPC64 ELF ABI.
          RelocationEntry REhst(SectionID,
                                StubTargetAddr - Section.Address + 2,
                                ELF::R_PPC64_ADDR16_HIGHEST, Value.Addend);
          RelocationEntry REhr(SectionID,
                               StubTargetAddr - Section.Address + 6,
                               ELF::R_PPC64_ADDR16_HIGHER, Value.Addend);
          RelocationEntry REh(SectionID,
                              StubTargetAddr - Section.Address + 14,
                              ELF::R_PPC64_ADDR16_HI, Value.Addend);
          RelocationEntry REl(SectionID,
                              StubTargetAddr - Section.Address + 18,
                              ELF::R_PPC64_ADDR16_LO, Value.Addend);

          if (Value.SymbolName) {
            addRelocationForSymbol(REhst, Value.SymbolName);
            addRelocationForSymbol(REhr,  Value.SymbolName);
            addRelocationForSymbol(REh,   Value.SymbolName);
            addRelocationForSymbol(REl,   Value.SymbolName);
          } else {
            addRelocationForSection(REhst, Value.SectionID);
            addRelocationForSection(REhr,  Value.SectionID);
            addRelocationForSection(REh,   Value.SectionID);
            addRelocationForSection(REl,   Value.SectionID);
          }

          resolveRelocation(Section, Offset,
                            (uint64_t)Section.Address + Section.StubOffset,
                            RelType, 0);
          if (SymType == SymbolRef::ST_Unknown)
            // Restore the TOC for external calls
            writeInt32BE(Target+4, 0xE8410028); // ld r2,40(r1)
          Section.StubOffset += getMaxStubSize();
        }
      }
    } else {
      RelocationEntry RE(SectionID, Offset, RelType, Value.Addend);
      // Extra check to avoid relocation againt empty symbols (usually
      // the R_PPC64_TOC).
      if (SymType != SymbolRef::ST_Unknown && TargetName.empty())
        Value.SymbolName = NULL;

      if (Value.SymbolName)
        addRelocationForSymbol(RE, Value.SymbolName);
      else
        addRelocationForSection(RE, Value.SectionID);
    }
  } else if (Arch == Triple::systemz &&
             (RelType == ELF::R_390_PLT32DBL ||
              RelType == ELF::R_390_GOTENT)) {
    // Create function stubs for both PLT and GOT references, regardless of
    // whether the GOT reference is to data or code.  The stub contains the
    // full address of the symbol, as needed by GOT references, and the
    // executable part only adds an overhead of 8 bytes.
    //
    // We could try to conserve space by allocating the code and data
    // parts of the stub separately.  However, as things stand, we allocate
    // a stub for every relocation, so using a GOT in JIT code should be
    // no less space efficient than using an explicit constant pool.
    DEBUG(dbgs() << "\t\tThis is a SystemZ indirect relocation.");
    SectionEntry &Section = Sections[SectionID];

    // Look for an existing stub.
    StubMap::const_iterator i = Stubs.find(Value);
    uintptr_t StubAddress;
    if (i != Stubs.end()) {
      StubAddress = uintptr_t(Section.Address) + i->second;
      DEBUG(dbgs() << " Stub function found\n");
    } else {
      // Create a new stub function.
      DEBUG(dbgs() << " Create a new stub function\n");

      uintptr_t BaseAddress = uintptr_t(Section.Address);
      uintptr_t StubAlignment = getStubAlignment();
      StubAddress = (BaseAddress + Section.StubOffset +
                     StubAlignment - 1) & -StubAlignment;
      unsigned StubOffset = StubAddress - BaseAddress;

      Stubs[Value] = StubOffset;
      createStubFunction((uint8_t *)StubAddress);
      RelocationEntry RE(SectionID, StubOffset + 8,
                         ELF::R_390_64, Value.Addend - Addend);
      if (Value.SymbolName)
        addRelocationForSymbol(RE, Value.SymbolName);
      else
        addRelocationForSection(RE, Value.SectionID);
      Section.StubOffset = StubOffset + getMaxStubSize();
    }

    if (RelType == ELF::R_390_GOTENT)
      resolveRelocation(Section, Offset, StubAddress + 8,
                        ELF::R_390_PC32DBL, Addend);
    else
      resolveRelocation(Section, Offset, StubAddress, RelType, Addend);
  } else if (Arch == Triple::x86_64 && RelType == ELF::R_X86_64_PLT32) {
    // The way the PLT relocations normally work is that the linker allocates the
    // PLT and this relocation makes a PC-relative call into the PLT.  The PLT
    // entry will then jump to an address provided by the GOT.  On first call, the
    // GOT address will point back into PLT code that resolves the symbol.  After
    // the first call, the GOT entry points to the actual function.
    //
    // For local functions we're ignoring all of that here and just replacing
    // the PLT32 relocation type with PC32, which will translate the relocation
    // into a PC-relative call directly to the function. For external symbols we
    // can't be sure the function will be within 2^32 bytes of the call site, so
    // we need to create a stub, which calls into the GOT.  This case is
    // equivalent to the usual PLT implementation except that we use the stub
    // mechanism in RuntimeDyld (which puts stubs at the end of the section)
    // rather than allocating a PLT section.
    if (Value.SymbolName) {
      // This is a call to an external function.
      // Look for an existing stub.
      SectionEntry &Section = Sections[SectionID];
      StubMap::const_iterator i = Stubs.find(Value);
      uintptr_t StubAddress;
      if (i != Stubs.end()) {
        StubAddress = uintptr_t(Section.Address) + i->second;
        DEBUG(dbgs() << " Stub function found\n");
      } else {
        // Create a new stub function (equivalent to a PLT entry).
        DEBUG(dbgs() << " Create a new stub function\n");

        uintptr_t BaseAddress = uintptr_t(Section.Address);
        uintptr_t StubAlignment = getStubAlignment();
        StubAddress = (BaseAddress + Section.StubOffset +
                      StubAlignment - 1) & -StubAlignment;
        unsigned StubOffset = StubAddress - BaseAddress;
        Stubs[Value] = StubOffset;
        createStubFunction((uint8_t *)StubAddress);

        // Create a GOT entry for the external function.
        GOTEntries.push_back(Value);

        // Make our stub function a relative call to the GOT entry.
        RelocationEntry RE(SectionID, StubOffset + 2,
                           ELF::R_X86_64_GOTPCREL, -4);
        addRelocationForSymbol(RE, Value.SymbolName);

        // Bump our stub offset counter
        Section.StubOffset = StubOffset + getMaxStubSize();
      }

      // Make the target call a call into the stub table.
      resolveRelocation(Section, Offset, StubAddress,
                      ELF::R_X86_64_PC32, Addend);
    } else {
      RelocationEntry RE(SectionID, Offset, ELF::R_X86_64_PC32, Value.Addend,
                         Value.Offset);
      addRelocationForSection(RE, Value.SectionID);
    }
  } else {
    if (Arch == Triple::x86_64 && RelType == ELF::R_X86_64_GOTPCREL) {
      GOTEntries.push_back(Value);
    }
    RelocationEntry RE(SectionID, Offset, RelType, Value.Addend, Value.Offset);
    if (Value.SymbolName)
      addRelocationForSymbol(RE, Value.SymbolName);
    else
      addRelocationForSection(RE, Value.SectionID);
  }
}

void RuntimeDyldELF::updateGOTEntries(StringRef Name, uint64_t Addr) {

  SmallVectorImpl<std::pair<SID, GOTRelocations> >::iterator it;
  SmallVectorImpl<std::pair<SID, GOTRelocations> >::iterator end = GOTs.end();

  for (it = GOTs.begin(); it != end; ++it) {
    GOTRelocations &GOTEntries = it->second;
    for (int i = 0, e = GOTEntries.size(); i != e; ++i) {
      if (GOTEntries[i].SymbolName != 0 && GOTEntries[i].SymbolName == Name) {
        GOTEntries[i].Offset = Addr;
      }
    }
  }
}

size_t RuntimeDyldELF::getGOTEntrySize() {
  // We don't use the GOT in all of these cases, but it's essentially free
  // to put them all here.
  size_t Result = 0;
  switch (Arch) {
  case Triple::x86_64:
  case Triple::aarch64:
  case Triple::ppc64:
  case Triple::ppc64le:
  case Triple::systemz:
    Result = sizeof(uint64_t);
    break;
  case Triple::x86:
  case Triple::arm:
  case Triple::thumb:
  case Triple::mips:
  case Triple::mipsel:
    Result = sizeof(uint32_t);
    break;
  default: llvm_unreachable("Unsupported CPU type!");
  }
  return Result;
}

uint64_t RuntimeDyldELF::findGOTEntry(uint64_t LoadAddress,
                                      uint64_t Offset) {

  const size_t GOTEntrySize = getGOTEntrySize();

  SmallVectorImpl<std::pair<SID, GOTRelocations> >::const_iterator it;
  SmallVectorImpl<std::pair<SID, GOTRelocations> >::const_iterator end = GOTs.end();

  int GOTIndex = -1;
  for (it = GOTs.begin(); it != end; ++it) {
    SID GOTSectionID = it->first;
    const GOTRelocations &GOTEntries = it->second;

    // Find the matching entry in our vector.
    uint64_t SymbolOffset = 0;
    for (int i = 0, e = GOTEntries.size(); i != e; ++i) {
      if (GOTEntries[i].SymbolName == 0) {
        if (getSectionLoadAddress(GOTEntries[i].SectionID) == LoadAddress &&
            GOTEntries[i].Offset == Offset) {
          GOTIndex = i;
          SymbolOffset = GOTEntries[i].Offset;
          break;
        }
      } else {
        // GOT entries for external symbols use the addend as the address when
        // the external symbol has been resolved.
        if (GOTEntries[i].Offset == LoadAddress) {
          GOTIndex = i;
          // Don't use the Addend here.  The relocation handler will use it.
          break;
        }
      }
    }

    if (GOTIndex != -1) {
      if (GOTEntrySize == sizeof(uint64_t)) {
        uint64_t *LocalGOTAddr = (uint64_t*)getSectionAddress(GOTSectionID);
        // Fill in this entry with the address of the symbol being referenced.
        LocalGOTAddr[GOTIndex] = LoadAddress + SymbolOffset;
      } else {
        uint32_t *LocalGOTAddr = (uint32_t*)getSectionAddress(GOTSectionID);
        // Fill in this entry with the address of the symbol being referenced.
        LocalGOTAddr[GOTIndex] = (uint32_t)(LoadAddress + SymbolOffset);
      }

      // Calculate the load address of this entry
      return getSectionLoadAddress(GOTSectionID) + (GOTIndex * GOTEntrySize);
    }
  }

  assert(GOTIndex != -1 && "Unable to find requested GOT entry.");
  return 0;
}

void RuntimeDyldELF::finalizeLoad(ObjSectionToIDMap &SectionMap) {
  // If necessary, allocate the global offset table
  if (MemMgr) {
    // Allocate the GOT if necessary
    size_t numGOTEntries = GOTEntries.size();
    if (numGOTEntries != 0) {
      // Allocate memory for the section
      unsigned SectionID = Sections.size();
      size_t TotalSize = numGOTEntries * getGOTEntrySize();
      uint8_t *Addr = MemMgr->allocateDataSection(TotalSize, getGOTEntrySize(),
                                                  SectionID, ".got", false);
      if (!Addr)
        report_fatal_error("Unable to allocate memory for GOT!");

      GOTs.push_back(std::make_pair(SectionID, GOTEntries));
      Sections.push_back(SectionEntry(".got", Addr, TotalSize, 0));
      // For now, initialize all GOT entries to zero.  We'll fill them in as
      // needed when GOT-based relocations are applied.
      memset(Addr, 0, TotalSize);
    }
  }
  else {
    report_fatal_error("Unable to allocate memory for GOT!");
  }

  // Look for and record the EH frame section.
  ObjSectionToIDMap::iterator i, e;
  for (i = SectionMap.begin(), e = SectionMap.end(); i != e; ++i) {
    const SectionRef &Section = i->first;
    StringRef Name;
    Section.getName(Name);
    if (Name == ".eh_frame") {
      UnregisteredEHFrameSections.push_back(i->second);
      break;
    }
  }
}

bool RuntimeDyldELF::isCompatibleFormat(const ObjectBuffer *Buffer) const {
  if (Buffer->getBufferSize() < strlen(ELF::ElfMagic))
    return false;
  return (memcmp(Buffer->getBufferStart(), ELF::ElfMagic, strlen(ELF::ElfMagic))) == 0;
}
} // namespace llvm
