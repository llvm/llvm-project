//===-- RuntimeDyld.cpp - Run-time dynamic linker for MC-JIT ----*- C++ -*-===//
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
#include "ObjectImageCommon.h"
#include "RuntimeDyldImpl.h"
#include "RuntimeDyldELF.h"
#include "RuntimeDyldMachO.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;
using namespace llvm::object;

// Empty out-of-line virtual destructor as the key function.
RTDyldMemoryManager::~RTDyldMemoryManager() {}
RuntimeDyldImpl::~RuntimeDyldImpl() {}

namespace llvm {

// Resolve the relocations for all symbols we currently know about.
void RuntimeDyldImpl::resolveRelocations() {
  // First, resolve relocations associated with external symbols.
  resolveExternalSymbols();

  // Just iterate over the sections we have and resolve all the relocations
  // in them. Gross overkill, but it gets the job done.
  for (int i = 0, e = Sections.size(); i != e; ++i) {
    uint64_t Addr = Sections[i].LoadAddress;
    DEBUG(dbgs() << "Resolving relocations Section #" << i
            << "\t" << format("%p", (uint8_t *)Addr)
            << "\n");
    resolveRelocationList(Relocations[i], Addr);
  }
}

void RuntimeDyldImpl::mapSectionAddress(const void *LocalAddress,
                                        uint64_t TargetAddress) {
  for (unsigned i = 0, e = Sections.size(); i != e; ++i) {
    if (Sections[i].Address == LocalAddress) {
      reassignSectionAddress(i, TargetAddress);
      return;
    }
  }
  llvm_unreachable("Attempting to remap address of unknown section!");
}

// Subclasses can implement this method to create specialized image instances.
// The caller owns the pointer that is returned.
ObjectImage *RuntimeDyldImpl::createObjectImage(ObjectBuffer *InputBuffer) {
  return new ObjectImageCommon(InputBuffer);
}

ObjectImage *RuntimeDyldImpl::loadObject(ObjectBuffer *InputBuffer) {
  OwningPtr<ObjectImage> obj(createObjectImage(InputBuffer));
  if (!obj)
    report_fatal_error("Unable to create object image from memory buffer!");

  Arch = (Triple::ArchType)obj->getArch();

  // Symbols found in this object
  StringMap<SymbolLoc> LocalSymbols;
  // Used sections from the object file
  ObjSectionToIDMap LocalSections;

  // Common symbols requiring allocation, with their sizes and alignments
  CommonSymbolMap CommonSymbols;
  // Maximum required total memory to allocate all common symbols
  uint64_t CommonSize = 0;

  error_code err;
  // Parse symbols
  DEBUG(dbgs() << "Parse symbols:\n");
  for (symbol_iterator i = obj->begin_symbols(), e = obj->end_symbols();
       i != e; i.increment(err)) {
    Check(err);
    object::SymbolRef::Type SymType;
    StringRef Name;
    Check(i->getType(SymType));
    Check(i->getName(Name));

    uint32_t flags;
    Check(i->getFlags(flags));

    bool isCommon = flags & SymbolRef::SF_Common;
    if (isCommon) {
      // Add the common symbols to a list.  We'll allocate them all below.
      uint64_t Align = getCommonSymbolAlignment(*i);
      uint64_t Size = 0;
      Check(i->getSize(Size));
      CommonSize += Size + Align;
      CommonSymbols[*i] = CommonSymbolInfo(Size, Align);
    } else {
      if (SymType == object::SymbolRef::ST_Function ||
          SymType == object::SymbolRef::ST_Data ||
          SymType == object::SymbolRef::ST_Unknown) {
        uint64_t FileOffset;
        StringRef SectionData;
        section_iterator si = obj->end_sections();
        Check(i->getFileOffset(FileOffset));
        Check(i->getSection(si));
        if (si == obj->end_sections()) continue;
        Check(si->getContents(SectionData));
        const uint8_t* SymPtr = (const uint8_t*)InputBuffer->getBufferStart() +
                                (uintptr_t)FileOffset;
        uintptr_t SectOffset = (uintptr_t)(SymPtr -
                                           (const uint8_t*)SectionData.begin());
        unsigned SectionID =
          findOrEmitSection(*obj,
                            *si,
                            SymType == object::SymbolRef::ST_Function,
                            LocalSections);
        LocalSymbols[Name.data()] = SymbolLoc(SectionID, SectOffset);
        DEBUG(dbgs() << "\tFileOffset: " << format("%p", (uintptr_t)FileOffset)
                     << " flags: " << flags
                     << " SID: " << SectionID
                     << " Offset: " << format("%p", SectOffset));
        bool isGlobal = flags & SymbolRef::SF_Global;
        if (isGlobal)
          GlobalSymbolTable[Name] = SymbolLoc(SectionID, SectOffset);
      }
    }
    DEBUG(dbgs() << "\tType: " << SymType << " Name: " << Name << "\n");
  }

  // Allocate common symbols
  if (CommonSize != 0)
    emitCommonSymbols(*obj, CommonSymbols, CommonSize, LocalSymbols);

  // Parse and process relocations
  DEBUG(dbgs() << "Parse relocations:\n");
  for (section_iterator si = obj->begin_sections(),
       se = obj->end_sections(); si != se; si.increment(err)) {
    Check(err);
    bool isFirstRelocation = true;
    unsigned SectionID = 0;
    StubMap Stubs;

    for (relocation_iterator i = si->begin_relocations(),
         e = si->end_relocations(); i != e; i.increment(err)) {
      Check(err);

      // If it's the first relocation in this section, find its SectionID
      if (isFirstRelocation) {
        SectionID = findOrEmitSection(*obj, *si, true, LocalSections);
        DEBUG(dbgs() << "\tSectionID: " << SectionID << "\n");
        isFirstRelocation = false;
      }

      ObjRelocationInfo RI;
      RI.SectionID = SectionID;
      Check(i->getAdditionalInfo(RI.AdditionalInfo));
      Check(i->getOffset(RI.Offset));
      Check(i->getSymbol(RI.Symbol));
      Check(i->getType(RI.Type));

      DEBUG(dbgs() << "\t\tAddend: " << RI.AdditionalInfo
                   << " Offset: " << format("%p", (uintptr_t)RI.Offset)
                   << " Type: " << (uint32_t)(RI.Type & 0xffffffffL)
                   << "\n");
      processRelocationRef(RI, *obj, LocalSections, LocalSymbols, Stubs);
    }
  }

  return obj.take();
}

void RuntimeDyldImpl::emitCommonSymbols(ObjectImage &Obj,
                                        const CommonSymbolMap &CommonSymbols,
                                        uint64_t TotalSize,
                                        SymbolTableMap &SymbolTable) {
  // Allocate memory for the section
  unsigned SectionID = Sections.size();
  uint8_t *Addr = MemMgr->allocateDataSection(TotalSize, sizeof(void*),
                                              SectionID);
  if (!Addr)
    report_fatal_error("Unable to allocate memory for common symbols!");
  uint64_t Offset = 0;
  Sections.push_back(SectionEntry(StringRef(), Addr, TotalSize, TotalSize, 0));
  memset(Addr, 0, TotalSize);

  DEBUG(dbgs() << "emitCommonSection SectionID: " << SectionID
               << " new addr: " << format("%p", Addr)
               << " DataSize: " << TotalSize
               << "\n");

  // Assign the address of each symbol
  for (CommonSymbolMap::const_iterator it = CommonSymbols.begin(),
       itEnd = CommonSymbols.end(); it != itEnd; it++) {
    uint64_t Size = it->second.first;
    uint64_t Align = it->second.second;
    StringRef Name;
    it->first.getName(Name);
    if (Align) {
      // This symbol has an alignment requirement.
      uint64_t AlignOffset = OffsetToAlignment((uint64_t)Addr, Align);
      Addr += AlignOffset;
      Offset += AlignOffset;
      DEBUG(dbgs() << "Allocating common symbol " << Name << " address " <<
                      format("%p\n", Addr));
    }
    Obj.updateSymbolAddress(it->first, (uint64_t)Addr);
    SymbolTable[Name.data()] = SymbolLoc(SectionID, Offset);
    Offset += Size;
    Addr += Size;
  }
}

unsigned RuntimeDyldImpl::emitSection(ObjectImage &Obj,
                                      const SectionRef &Section,
                                      bool IsCode) {

  unsigned StubBufSize = 0,
           StubSize = getMaxStubSize();
  error_code err;
  if (StubSize > 0) {
    for (relocation_iterator i = Section.begin_relocations(),
         e = Section.end_relocations(); i != e; i.increment(err), Check(err))
      StubBufSize += StubSize;
  }
  StringRef data;
  uint64_t Alignment64;
  Check(Section.getContents(data));
  Check(Section.getAlignment(Alignment64));

  unsigned Alignment = (unsigned)Alignment64 & 0xffffffffL;
  bool IsRequired;
  bool IsVirtual;
  bool IsZeroInit;
  uint64_t DataSize;
  StringRef Name;
  Check(Section.isRequiredForExecution(IsRequired));
  Check(Section.isVirtual(IsVirtual));
  Check(Section.isZeroInit(IsZeroInit));
  Check(Section.getSize(DataSize));
  Check(Section.getName(Name));

  unsigned Allocate;
  unsigned SectionID = Sections.size();
  uint8_t *Addr;
  const char *pData = 0;

  // Some sections, such as debug info, don't need to be loaded for execution.
  // Leave those where they are.
  if (IsRequired) {
    Allocate = DataSize + StubBufSize;
    Addr = IsCode
      ? MemMgr->allocateCodeSection(Allocate, Alignment, SectionID)
      : MemMgr->allocateDataSection(Allocate, Alignment, SectionID);
    if (!Addr)
      report_fatal_error("Unable to allocate section memory!");

    // Virtual sections have no data in the object image, so leave pData = 0
    if (!IsVirtual)
      pData = data.data();

    // Zero-initialize or copy the data from the image
    if (IsZeroInit || IsVirtual)
      memset(Addr, 0, DataSize);
    else
      memcpy(Addr, pData, DataSize);

    DEBUG(dbgs() << "emitSection SectionID: " << SectionID
                 << " Name: " << Name
                 << " obj addr: " << format("%p", pData)
                 << " new addr: " << format("%p", Addr)
                 << " DataSize: " << DataSize
                 << " StubBufSize: " << StubBufSize
                 << " Allocate: " << Allocate
                 << "\n");
    Obj.updateSectionAddress(Section, (uint64_t)Addr);
  }
  else {
    // Even if we didn't load the section, we need to record an entry for it
    // to handle later processing (and by 'handle' I mean don't do anything
    // with these sections).
    Allocate = 0;
    Addr = 0;
    DEBUG(dbgs() << "emitSection SectionID: " << SectionID
                 << " Name: " << Name
                 << " obj addr: " << format("%p", data.data())
                 << " new addr: 0"
                 << " DataSize: " << DataSize
                 << " StubBufSize: " << StubBufSize
                 << " Allocate: " << Allocate
                 << "\n");
  }

  Sections.push_back(SectionEntry(Name, Addr, Allocate, DataSize,
				  (uintptr_t)pData));
  return SectionID;
}

unsigned RuntimeDyldImpl::findOrEmitSection(ObjectImage &Obj,
                                            const SectionRef &Section,
                                            bool IsCode,
                                            ObjSectionToIDMap &LocalSections) {

  unsigned SectionID = 0;
  ObjSectionToIDMap::iterator i = LocalSections.find(Section);
  if (i != LocalSections.end())
    SectionID = i->second;
  else {
    SectionID = emitSection(Obj, Section, IsCode);
    LocalSections[Section] = SectionID;
  }
  return SectionID;
}

void RuntimeDyldImpl::addRelocationForSection(const RelocationEntry &RE,
                                              unsigned SectionID) {
  Relocations[SectionID].push_back(RE);
}

void RuntimeDyldImpl::addRelocationForSymbol(const RelocationEntry &RE,
                                             StringRef SymbolName) {
  // Relocation by symbol.  If the symbol is found in the global symbol table,
  // create an appropriate section relocation.  Otherwise, add it to
  // ExternalSymbolRelocations.
  SymbolTableMap::const_iterator Loc =
      GlobalSymbolTable.find(SymbolName);
  if (Loc == GlobalSymbolTable.end()) {
    ExternalSymbolRelocations[SymbolName].push_back(RE);
  } else {
    // Copy the RE since we want to modify its addend.
    RelocationEntry RECopy = RE;
    RECopy.Addend += Loc->second.second;
    Relocations[Loc->second.first].push_back(RECopy);
  }
}

uint8_t *RuntimeDyldImpl::createStubFunction(uint8_t *Addr) {
  if (Arch == Triple::arm) {
    // TODO: There is only ARM far stub now. We should add the Thumb stub,
    // and stubs for branches Thumb - ARM and ARM - Thumb.
    uint32_t *StubAddr = (uint32_t*)Addr;
    *StubAddr = 0xe51ff004; // ldr pc,<label>
    return (uint8_t*)++StubAddr;
  } else if (Arch == Triple::mipsel) {
    uint32_t *StubAddr = (uint32_t*)Addr;
    // 0:   3c190000        lui     t9,%hi(addr).
    // 4:   27390000        addiu   t9,t9,%lo(addr).
    // 8:   03200008        jr      t9.
    // c:   00000000        nop.
    const unsigned LuiT9Instr = 0x3c190000, AdduiT9Instr = 0x27390000;
    const unsigned JrT9Instr = 0x03200008, NopInstr = 0x0;

    *StubAddr = LuiT9Instr;
    StubAddr++;
    *StubAddr = AdduiT9Instr;
    StubAddr++;
    *StubAddr = JrT9Instr;
    StubAddr++;
    *StubAddr = NopInstr;
    return Addr;
  } else if (Arch == Triple::ppc64) {
    // PowerPC64 stub: the address points to a function descriptor
    // instead of the function itself. Load the function address
    // on r11 and sets it to control register. Also loads the function
    // TOC in r2 and environment pointer to r11.
    writeInt32BE(Addr,    0x3D800000); // lis   r12, highest(addr)
    writeInt32BE(Addr+4,  0x618C0000); // ori   r12, higher(addr)
    writeInt32BE(Addr+8,  0x798C07C6); // sldi  r12, r12, 32
    writeInt32BE(Addr+12, 0x658C0000); // oris  r12, r12, h(addr)
    writeInt32BE(Addr+16, 0x618C0000); // ori   r12, r12, l(addr)
    writeInt32BE(Addr+20, 0xF8410028); // std   r2,  40(r1)
    writeInt32BE(Addr+24, 0xE96C0000); // ld    r11, 0(r12)
    writeInt32BE(Addr+28, 0xE84C0008); // ld    r2,  0(r12)
    writeInt32BE(Addr+32, 0x7D6903A6); // mtctr r11
    writeInt32BE(Addr+36, 0xE96C0010); // ld    r11, 16(r2)
    writeInt32BE(Addr+40, 0x4E800420); // bctr

    return Addr;
  }
  return Addr;
}

// Assign an address to a symbol name and resolve all the relocations
// associated with it.
void RuntimeDyldImpl::reassignSectionAddress(unsigned SectionID,
                                             uint64_t Addr) {
  // The address to use for relocation resolution is not
  // the address of the local section buffer. We must be doing
  // a remote execution environment of some sort. Relocations can't
  // be applied until all the sections have been moved.  The client must
  // trigger this with a call to MCJIT::finalize() or
  // RuntimeDyld::resolveRelocations().
  //
  // Addr is a uint64_t because we can't assume the pointer width
  // of the target is the same as that of the host. Just use a generic
  // "big enough" type.
  Sections[SectionID].LoadAddress = Addr;
}

void RuntimeDyldImpl::resolveRelocationEntry(const RelocationEntry &RE,
                                             uint64_t Value) {
  // Ignore relocations for sections that were not loaded
  if (Sections[RE.SectionID].Address != 0) {
    DEBUG(dbgs() << "\tSectionID: " << RE.SectionID
          << " + " << RE.Offset << " ("
          << format("%p", Sections[RE.SectionID].Address + RE.Offset) << ")"
          << " RelType: " << RE.RelType
          << " Addend: " << RE.Addend
          << "\n");

    resolveRelocation(Sections[RE.SectionID], RE.Offset,
                      Value, RE.RelType, RE.Addend);
  }
}

void RuntimeDyldImpl::resolveRelocationList(const RelocationList &Relocs,
                                            uint64_t Value) {
  for (unsigned i = 0, e = Relocs.size(); i != e; ++i) {
    resolveRelocationEntry(Relocs[i], Value);
  }
}

void RuntimeDyldImpl::resolveExternalSymbols() {
  StringMap<RelocationList>::iterator i = ExternalSymbolRelocations.begin(),
                                      e = ExternalSymbolRelocations.end();
  for (; i != e; i++) {
    StringRef Name = i->first();
    RelocationList &Relocs = i->second;
    SymbolTableMap::const_iterator Loc = GlobalSymbolTable.find(Name);
    if (Loc == GlobalSymbolTable.end()) {
      // This is an external symbol, try to get it address from
      // MemoryManager.
      uint8_t *Addr = (uint8_t*) MemMgr->getPointerToNamedFunction(Name.data(),
                                                                   true);
      DEBUG(dbgs() << "Resolving relocations Name: " << Name
              << "\t" << format("%p", Addr)
              << "\n");
      resolveRelocationList(Relocs, (uintptr_t)Addr);
    } else {
      report_fatal_error("Expected external symbol");
    }
  }
}


//===----------------------------------------------------------------------===//
// RuntimeDyld class implementation
RuntimeDyld::RuntimeDyld(RTDyldMemoryManager *mm) {
  Dyld = 0;
  MM = mm;
}

RuntimeDyld::~RuntimeDyld() {
  delete Dyld;
}

ObjectImage *RuntimeDyld::loadObject(ObjectBuffer *InputBuffer) {
  if (!Dyld) {
    sys::LLVMFileType type = sys::IdentifyFileType(
            InputBuffer->getBufferStart(),
            static_cast<unsigned>(InputBuffer->getBufferSize()));
    switch (type) {
      case sys::ELF_Relocatable_FileType:
      case sys::ELF_Executable_FileType:
      case sys::ELF_SharedObject_FileType:
      case sys::ELF_Core_FileType:
        Dyld = new RuntimeDyldELF(MM);
        break;
      case sys::Mach_O_Object_FileType:
      case sys::Mach_O_Executable_FileType:
      case sys::Mach_O_FixedVirtualMemorySharedLib_FileType:
      case sys::Mach_O_Core_FileType:
      case sys::Mach_O_PreloadExecutable_FileType:
      case sys::Mach_O_DynamicallyLinkedSharedLib_FileType:
      case sys::Mach_O_DynamicLinker_FileType:
      case sys::Mach_O_Bundle_FileType:
      case sys::Mach_O_DynamicallyLinkedSharedLibStub_FileType:
      case sys::Mach_O_DSYMCompanion_FileType:
        Dyld = new RuntimeDyldMachO(MM);
        break;
      case sys::Unknown_FileType:
      case sys::Bitcode_FileType:
      case sys::Archive_FileType:
      case sys::COFF_FileType:
        report_fatal_error("Incompatible object format!");
    }
  } else {
    if (!Dyld->isCompatibleFormat(InputBuffer))
      report_fatal_error("Incompatible object format!");
  }

  return Dyld->loadObject(InputBuffer);
}

void *RuntimeDyld::getSymbolAddress(StringRef Name) {
  return Dyld->getSymbolAddress(Name);
}

uint64_t RuntimeDyld::getSymbolLoadAddress(StringRef Name) {
  return Dyld->getSymbolLoadAddress(Name);
}

void RuntimeDyld::resolveRelocations() {
  Dyld->resolveRelocations();
}

void RuntimeDyld::reassignSectionAddress(unsigned SectionID,
                                         uint64_t Addr) {
  Dyld->reassignSectionAddress(SectionID, Addr);
}

void RuntimeDyld::mapSectionAddress(const void *LocalAddress,
                                    uint64_t TargetAddress) {
  Dyld->mapSectionAddress(LocalAddress, TargetAddress);
}

StringRef RuntimeDyld::getErrorString() {
  return Dyld->getErrorString();
}

} // end namespace llvm
