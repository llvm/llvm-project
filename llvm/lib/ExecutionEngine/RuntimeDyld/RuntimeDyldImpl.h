//===-- RuntimeDyldImpl.h - Run-time dynamic linker for MC-JIT --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface for the implementations of runtime dynamic linker facilities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_RUNTIME_DYLD_IMPL_H
#define LLVM_RUNTIME_DYLD_IMPL_H

#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ExecutionEngine/ObjectImage.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/SwapByteOrder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include <map>

using namespace llvm;
using namespace llvm::object;

namespace llvm {

class ObjectBuffer;
class Twine;


/// SectionEntry - represents a section emitted into memory by the dynamic
/// linker.
class SectionEntry {
public:
  /// Name - section name.
  StringRef Name;

  /// Address - address in the linker's memory where the section resides.
  uint8_t *Address;

  /// Size - section size.
  size_t Size;

  /// LoadAddress - the address of the section in the target process's memory.
  /// Used for situations in which JIT-ed code is being executed in the address
  /// space of a separate process.  If the code executes in the same address
  /// space where it was JIT-ed, this just equals Address.
  uint64_t LoadAddress;

  /// StubOffset - used for architectures with stub functions for far
  /// relocations (like ARM).
  uintptr_t StubOffset;

  /// ObjAddress - address of the section in the in-memory object file.  Used
  /// for calculating relocations in some object formats (like MachO).
  uintptr_t ObjAddress;

  SectionEntry(StringRef name, uint8_t *address, size_t size,
	       uintptr_t stubOffset, uintptr_t objAddress)
    : Name(name), Address(address), Size(size), LoadAddress((uintptr_t)address),
      StubOffset(stubOffset), ObjAddress(objAddress) {}
};

/// RelocationEntry - used to represent relocations internally in the dynamic
/// linker.
class RelocationEntry {
public:
  /// SectionID - the section this relocation points to.
  unsigned SectionID;

  /// Offset - offset into the section.
  uintptr_t Offset;

  /// RelType - relocation type.
  uint32_t RelType;

  /// Addend - the relocation addend encoded in the instruction itself.  Also
  /// used to make a relocation section relative instead of symbol relative.
  intptr_t Addend;

  RelocationEntry(unsigned id, uint64_t offset, uint32_t type, int64_t addend)
    : SectionID(id), Offset(offset), RelType(type), Addend(addend) {}
};

/// ObjRelocationInfo - relocation information as read from the object file.
/// Used to pass around data taken from object::RelocationRef, together with
/// the section to which the relocation points (represented by a SectionID).
class ObjRelocationInfo {
public:
  unsigned  SectionID;
  uint64_t  Offset;
  SymbolRef Symbol;
  uint64_t  Type;
  int64_t   AdditionalInfo;
};

class RelocationValueRef {
public:
  unsigned  SectionID;
  intptr_t  Addend;
  const char *SymbolName;
  RelocationValueRef(): SectionID(0), Addend(0), SymbolName(0) {}

  inline bool operator==(const RelocationValueRef &Other) const {
    return std::memcmp(this, &Other, sizeof(RelocationValueRef)) == 0;
  }
  inline bool operator <(const RelocationValueRef &Other) const {
    return std::memcmp(this, &Other, sizeof(RelocationValueRef)) < 0;
  }
};

class RuntimeDyldImpl {
protected:
  // The MemoryManager to load objects into.
  RTDyldMemoryManager *MemMgr;

  // A list of all sections emitted by the dynamic linker.  These sections are
  // referenced in the code by means of their index in this list - SectionID.
  typedef SmallVector<SectionEntry, 64> SectionList;
  SectionList Sections;

  // Keep a map of sections from object file to the SectionID which
  // references it.
  typedef std::map<SectionRef, unsigned> ObjSectionToIDMap;

  // A global symbol table for symbols from all loaded modules.  Maps the
  // symbol name to a (SectionID, offset in section) pair.
  typedef std::pair<unsigned, uintptr_t> SymbolLoc;
  typedef StringMap<SymbolLoc> SymbolTableMap;
  SymbolTableMap GlobalSymbolTable;

  // Pair representing the size and alignment requirement for a common symbol.
  typedef std::pair<unsigned, unsigned> CommonSymbolInfo;
  // Keep a map of common symbols to their info pairs
  typedef std::map<SymbolRef, CommonSymbolInfo> CommonSymbolMap;

  // For each symbol, keep a list of relocations based on it. Anytime
  // its address is reassigned (the JIT re-compiled the function, e.g.),
  // the relocations get re-resolved.
  // The symbol (or section) the relocation is sourced from is the Key
  // in the relocation list where it's stored.
  typedef SmallVector<RelocationEntry, 64> RelocationList;
  // Relocations to sections already loaded. Indexed by SectionID which is the
  // source of the address. The target where the address will be written is
  // SectionID/Offset in the relocation itself.
  DenseMap<unsigned, RelocationList> Relocations;

  // Relocations to external symbols that are not yet resolved.  Symbols are
  // external when they aren't found in the global symbol table of all loaded
  // modules.  This map is indexed by symbol name.
  StringMap<RelocationList> ExternalSymbolRelocations;

  typedef std::map<RelocationValueRef, uintptr_t> StubMap;

  Triple::ArchType Arch;

  inline unsigned getMaxStubSize() {
    if (Arch == Triple::arm || Arch == Triple::thumb)
      return 8; // 32-bit instruction and 32-bit address
    else if (Arch == Triple::mipsel)
      return 16;
    else if (Arch == Triple::ppc64)
      return 44;
    else
      return 0;
  }

  bool HasError;
  std::string ErrorStr;

  // Set the error state and record an error string.
  bool Error(const Twine &Msg) {
    ErrorStr = Msg.str();
    HasError = true;
    return true;
  }

  uint64_t getSectionLoadAddress(unsigned SectionID) {
    return Sections[SectionID].LoadAddress;
  }

  uint8_t *getSectionAddress(unsigned SectionID) {
    return (uint8_t*)Sections[SectionID].Address;
  }

  // Subclasses can override this method to get the alignment requirement of
  // a common symbol. Returns no alignment requirement if not implemented.
  virtual unsigned getCommonSymbolAlignment(const SymbolRef &Sym) {
    return 0;
  }


  void writeInt16BE(uint8_t *Addr, uint16_t Value) {
    if (sys::isLittleEndianHost())
      Value = sys::SwapByteOrder(Value);
    *Addr     = (Value >> 8) & 0xFF;
    *(Addr+1) = Value & 0xFF;
  }

  void writeInt32BE(uint8_t *Addr, uint32_t Value) {
    if (sys::isLittleEndianHost())
      Value = sys::SwapByteOrder(Value);
    *Addr     = (Value >> 24) & 0xFF;
    *(Addr+1) = (Value >> 16) & 0xFF;
    *(Addr+2) = (Value >> 8) & 0xFF;
    *(Addr+3) = Value & 0xFF;
  }

  void writeInt64BE(uint8_t *Addr, uint64_t Value) {
    if (sys::isLittleEndianHost())
      Value = sys::SwapByteOrder(Value);
    *Addr     = (Value >> 56) & 0xFF;
    *(Addr+1) = (Value >> 48) & 0xFF;
    *(Addr+2) = (Value >> 40) & 0xFF;
    *(Addr+3) = (Value >> 32) & 0xFF;
    *(Addr+4) = (Value >> 24) & 0xFF;
    *(Addr+5) = (Value >> 16) & 0xFF;
    *(Addr+6) = (Value >> 8) & 0xFF;
    *(Addr+7) = Value & 0xFF;
  }

  /// \brief Given the common symbols discovered in the object file, emit a
  /// new section for them and update the symbol mappings in the object and
  /// symbol table.
  void emitCommonSymbols(ObjectImage &Obj,
                         const CommonSymbolMap &CommonSymbols,
                         uint64_t TotalSize,
                         SymbolTableMap &SymbolTable);

  /// \brief Emits section data from the object file to the MemoryManager.
  /// \param IsCode if it's true then allocateCodeSection() will be
  ///        used for emits, else allocateDataSection() will be used.
  /// \return SectionID.
  unsigned emitSection(ObjectImage &Obj,
                       const SectionRef &Section,
                       bool IsCode);

  /// \brief Find Section in LocalSections. If the secton is not found - emit
  ///        it and store in LocalSections.
  /// \param IsCode if it's true then allocateCodeSection() will be
  ///        used for emmits, else allocateDataSection() will be used.
  /// \return SectionID.
  unsigned findOrEmitSection(ObjectImage &Obj,
                             const SectionRef &Section,
                             bool IsCode,
                             ObjSectionToIDMap &LocalSections);

  // \brief Add a relocation entry that uses the given section.
  void addRelocationForSection(const RelocationEntry &RE, unsigned SectionID);

  // \brief Add a relocation entry that uses the given symbol.  This symbol may
  // be found in the global symbol table, or it may be external.
  void addRelocationForSymbol(const RelocationEntry &RE, StringRef SymbolName);

  /// \brief Emits long jump instruction to Addr.
  /// \return Pointer to the memory area for emitting target address.
  uint8_t* createStubFunction(uint8_t *Addr);

  /// \brief Resolves relocations from Relocs list with address from Value.
  void resolveRelocationList(const RelocationList &Relocs, uint64_t Value);
  void resolveRelocationEntry(const RelocationEntry &RE, uint64_t Value);

  /// \brief A object file specific relocation resolver
  /// \param Section The section where the relocation is being applied
  /// \param Offset The offset into the section for this relocation
  /// \param Value Target symbol address to apply the relocation action
  /// \param Type object file specific relocation type
  /// \param Addend A constant addend used to compute the value to be stored
  ///        into the relocatable field
  virtual void resolveRelocation(const SectionEntry &Section,
                                 uint64_t Offset,
                                 uint64_t Value,
                                 uint32_t Type,
                                 int64_t Addend) = 0;

  /// \brief Parses the object file relocation and stores it to Relocations
  ///        or SymbolRelocations (this depends on the object file type).
  virtual void processRelocationRef(const ObjRelocationInfo &Rel,
                                    ObjectImage &Obj,
                                    ObjSectionToIDMap &ObjSectionToID,
                                    const SymbolTableMap &Symbols,
                                    StubMap &Stubs) = 0;

  /// \brief Resolve relocations to external symbols.
  void resolveExternalSymbols();
  virtual ObjectImage *createObjectImage(ObjectBuffer *InputBuffer);
public:
  RuntimeDyldImpl(RTDyldMemoryManager *mm) : MemMgr(mm), HasError(false) {}

  virtual ~RuntimeDyldImpl();

  ObjectImage *loadObject(ObjectBuffer *InputBuffer);

  void *getSymbolAddress(StringRef Name) {
    // FIXME: Just look up as a function for now. Overly simple of course.
    // Work in progress.
    if (GlobalSymbolTable.find(Name) == GlobalSymbolTable.end())
      return 0;
    SymbolLoc Loc = GlobalSymbolTable.lookup(Name);
    return getSectionAddress(Loc.first) + Loc.second;
  }

  uint64_t getSymbolLoadAddress(StringRef Name) {
    // FIXME: Just look up as a function for now. Overly simple of course.
    // Work in progress.
    if (GlobalSymbolTable.find(Name) == GlobalSymbolTable.end())
      return 0;
    SymbolLoc Loc = GlobalSymbolTable.lookup(Name);
    return getSectionLoadAddress(Loc.first) + Loc.second;
  }

  void resolveRelocations();

  void reassignSectionAddress(unsigned SectionID, uint64_t Addr);

  void mapSectionAddress(const void *LocalAddress, uint64_t TargetAddress);

  // Is the linker in an error state?
  bool hasError() { return HasError; }

  // Mark the error condition as handled and continue.
  void clearError() { HasError = false; }

  // Get the error message.
  StringRef getErrorString() { return ErrorStr; }

  virtual bool isCompatibleFormat(const ObjectBuffer *Buffer) const = 0;
};

} // end namespace llvm


#endif
