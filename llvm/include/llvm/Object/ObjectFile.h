//===- ObjectFile.h - File format independent object file -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares a file format independent ObjectFile class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_OBJECT_FILE_H
#define LLVM_OBJECT_OBJECT_FILE_H

#include "llvm/Object/Binary.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstring>
#include <vector>

namespace llvm {
namespace object {

class ObjectFile;

union DataRefImpl {
  struct {
    // ELF needs this for relocations. This entire union should probably be a
    // char[max(8, sizeof(uintptr_t))] and require the impl to cast.
    uint16_t a, b;
    uint32_t c;
  } w;
  struct {
    uint32_t a, b;
  } d;
  uintptr_t p;
  DataRefImpl() {
    std::memset(this, 0, sizeof(DataRefImpl));
  }
};

template<class content_type>
class content_iterator {
  content_type Current;
public:
  content_iterator(content_type symb)
    : Current(symb) {}

  const content_type* operator->() const {
    return &Current;
  }

  const content_type &operator*() const {
    return Current;
  }

  bool operator==(const content_iterator &other) const {
    return Current == other.Current;
  }

  bool operator!=(const content_iterator &other) const {
    return !(*this == other);
  }

  content_iterator& increment(error_code &err) {
    content_type next;
    if (error_code ec = Current.getNext(next))
      err = ec;
    else
      Current = next;
    return *this;
  }
};

inline bool operator==(const DataRefImpl &a, const DataRefImpl &b) {
  // Check bitwise identical. This is the only legal way to compare a union w/o
  // knowing which member is in use.
  return std::memcmp(&a, &b, sizeof(DataRefImpl)) == 0;
}

inline bool operator<(const DataRefImpl &a, const DataRefImpl &b) {
  // Check bitwise identical. This is the only legal way to compare a union w/o
  // knowing which member is in use.
  return std::memcmp(&a, &b, sizeof(DataRefImpl)) < 0;
}

class SymbolRef;

/// RelocationRef - This is a value type class that represents a single
/// relocation in the list of relocations in the object file.
class RelocationRef {
  DataRefImpl RelocationPimpl;
  const ObjectFile *OwningObject;

public:
  RelocationRef() : OwningObject(NULL) { }

  RelocationRef(DataRefImpl RelocationP, const ObjectFile *Owner);

  bool operator==(const RelocationRef &Other) const;

  error_code getNext(RelocationRef &Result) const;

  error_code getAddress(uint64_t &Result) const;
  error_code getOffset(uint64_t &Result) const;
  error_code getSymbol(SymbolRef &Result) const;
  error_code getType(uint64_t &Result) const;

  /// @brief Indicates whether this relocation should hidden when listing
  /// relocations, usually because it is the trailing part of a multipart
  /// relocation that will be printed as part of the leading relocation.
  error_code getHidden(bool &Result) const;

  /// @brief Get a string that represents the type of this relocation.
  ///
  /// This is for display purposes only.
  error_code getTypeName(SmallVectorImpl<char> &Result) const;
  error_code getAdditionalInfo(int64_t &Result) const;

  /// @brief Get a string that represents the calculation of the value of this
  ///        relocation.
  ///
  /// This is for display purposes only.
  error_code getValueString(SmallVectorImpl<char> &Result) const;

  DataRefImpl getRawDataRefImpl() const;
};
typedef content_iterator<RelocationRef> relocation_iterator;

/// SectionRef - This is a value type class that represents a single section in
/// the list of sections in the object file.
class SectionRef {
  friend class SymbolRef;
  DataRefImpl SectionPimpl;
  const ObjectFile *OwningObject;

public:
  SectionRef() : OwningObject(NULL) { }

  SectionRef(DataRefImpl SectionP, const ObjectFile *Owner);

  bool operator==(const SectionRef &Other) const;
  bool operator<(const SectionRef &Other) const;

  error_code getNext(SectionRef &Result) const;

  error_code getName(StringRef &Result) const;
  error_code getAddress(uint64_t &Result) const;
  error_code getSize(uint64_t &Result) const;
  error_code getContents(StringRef &Result) const;

  /// @brief Get the alignment of this section as the actual value (not log 2).
  error_code getAlignment(uint64_t &Result) const;

  // FIXME: Move to the normalization layer when it's created.
  error_code isText(bool &Result) const;
  error_code isData(bool &Result) const;
  error_code isBSS(bool &Result) const;
  error_code isRequiredForExecution(bool &Result) const;
  error_code isVirtual(bool &Result) const;
  error_code isZeroInit(bool &Result) const;
  error_code isReadOnlyData(bool &Result) const;

  error_code containsSymbol(SymbolRef S, bool &Result) const;

  relocation_iterator begin_relocations() const;
  relocation_iterator end_relocations() const;

  DataRefImpl getRawDataRefImpl() const;
};
typedef content_iterator<SectionRef> section_iterator;

/// SymbolRef - This is a value type class that represents a single symbol in
/// the list of symbols in the object file.
class SymbolRef {
  friend class SectionRef;
  DataRefImpl SymbolPimpl;
  const ObjectFile *OwningObject;

public:
  SymbolRef() : OwningObject(NULL) { }

  enum Type {
    ST_Unknown, // Type not specified
    ST_Data,
    ST_Debug,
    ST_File,
    ST_Function,
    ST_Other
  };

  enum Flags {
    SF_None            = 0,
    SF_Undefined       = 1U << 0,  // Symbol is defined in another object file
    SF_Global          = 1U << 1,  // Global symbol
    SF_Weak            = 1U << 2,  // Weak symbol
    SF_Absolute        = 1U << 3,  // Absolute symbol
    SF_ThreadLocal     = 1U << 4,  // Thread local symbol
    SF_Common          = 1U << 5,  // Symbol has common linkage
    SF_FormatSpecific  = 1U << 31  // Specific to the object file format
                                   // (e.g. section symbols)
  };

  SymbolRef(DataRefImpl SymbolP, const ObjectFile *Owner);

  bool operator==(const SymbolRef &Other) const;
  bool operator<(const SymbolRef &Other) const;

  error_code getNext(SymbolRef &Result) const;

  error_code getName(StringRef &Result) const;
  /// Returns the symbol virtual address (i.e. address at which it will be
  /// mapped).
  error_code getAddress(uint64_t &Result) const;
  error_code getFileOffset(uint64_t &Result) const;
  error_code getSize(uint64_t &Result) const;
  error_code getType(SymbolRef::Type &Result) const;

  /// Returns the ascii char that should be displayed in a symbol table dump via
  /// nm for this symbol.
  error_code getNMTypeChar(char &Result) const;

  /// Get symbol flags (bitwise OR of SymbolRef::Flags)
  error_code getFlags(uint32_t &Result) const;

  /// @brief Return true for common symbols such as uninitialized globals
  error_code isCommon(bool &Result) const;

  /// @brief Get section this symbol is defined in reference to. Result is
  /// end_sections() if it is undefined or is an absolute symbol.
  error_code getSection(section_iterator &Result) const;

  /// @brief Get value of the symbol in the symbol table.
  error_code getValue(uint64_t &Val) const;

  DataRefImpl getRawDataRefImpl() const;
};
typedef content_iterator<SymbolRef> symbol_iterator;

/// LibraryRef - This is a value type class that represents a single library in
/// the list of libraries needed by a shared or dynamic object.
class LibraryRef {
  friend class SectionRef;
  DataRefImpl LibraryPimpl;
  const ObjectFile *OwningObject;

public:
  LibraryRef() : OwningObject(NULL) { }

  LibraryRef(DataRefImpl LibraryP, const ObjectFile *Owner);

  bool operator==(const LibraryRef &Other) const;
  bool operator<(const LibraryRef &Other) const;

  error_code getNext(LibraryRef &Result) const;

  // Get the path to this library, as stored in the object file.
  error_code getPath(StringRef &Result) const;

  DataRefImpl getRawDataRefImpl() const;
};
typedef content_iterator<LibraryRef> library_iterator;

const uint64_t UnknownAddressOrSize = ~0ULL;

/// ObjectFile - This class is the base class for all object file types.
/// Concrete instances of this object are created by createObjectFile, which
/// figures out which type to create.
class ObjectFile : public Binary {
  virtual void anchor();
  ObjectFile() LLVM_DELETED_FUNCTION;
  ObjectFile(const ObjectFile &other) LLVM_DELETED_FUNCTION;

protected:
  ObjectFile(unsigned int Type, MemoryBuffer *source, error_code &ec);

  const uint8_t *base() const {
    return reinterpret_cast<const uint8_t *>(Data->getBufferStart());
  }

  // These functions are for SymbolRef to call internally. The main goal of
  // this is to allow SymbolRef::SymbolPimpl to point directly to the symbol
  // entry in the memory mapped object file. SymbolPimpl cannot contain any
  // virtual functions because then it could not point into the memory mapped
  // file.
  //
  // Implementations assume that the DataRefImpl is valid and has not been
  // modified externally. It's UB otherwise.
  friend class SymbolRef;
  virtual error_code getSymbolNext(DataRefImpl Symb, SymbolRef &Res) const = 0;
  virtual error_code getSymbolName(DataRefImpl Symb, StringRef &Res) const = 0;
  virtual error_code getSymbolAddress(DataRefImpl Symb, uint64_t &Res) const = 0;
  virtual error_code getSymbolFileOffset(DataRefImpl Symb, uint64_t &Res)const=0;
  virtual error_code getSymbolSize(DataRefImpl Symb, uint64_t &Res) const = 0;
  virtual error_code getSymbolType(DataRefImpl Symb,
                                   SymbolRef::Type &Res) const = 0;
  virtual error_code getSymbolNMTypeChar(DataRefImpl Symb, char &Res) const = 0;
  virtual error_code getSymbolFlags(DataRefImpl Symb,
                                    uint32_t &Res) const = 0;
  virtual error_code getSymbolSection(DataRefImpl Symb,
                                      section_iterator &Res) const = 0;
  virtual error_code getSymbolValue(DataRefImpl Symb, uint64_t &Val) const = 0;

  // Same as above for SectionRef.
  friend class SectionRef;
  virtual error_code getSectionNext(DataRefImpl Sec, SectionRef &Res) const = 0;
  virtual error_code getSectionName(DataRefImpl Sec, StringRef &Res) const = 0;
  virtual error_code getSectionAddress(DataRefImpl Sec, uint64_t &Res) const =0;
  virtual error_code getSectionSize(DataRefImpl Sec, uint64_t &Res) const = 0;
  virtual error_code getSectionContents(DataRefImpl Sec, StringRef &Res)const=0;
  virtual error_code getSectionAlignment(DataRefImpl Sec, uint64_t &Res)const=0;
  virtual error_code isSectionText(DataRefImpl Sec, bool &Res) const = 0;
  virtual error_code isSectionData(DataRefImpl Sec, bool &Res) const = 0;
  virtual error_code isSectionBSS(DataRefImpl Sec, bool &Res) const = 0;
  virtual error_code isSectionRequiredForExecution(DataRefImpl Sec,
                                                   bool &Res) const = 0;
  // A section is 'virtual' if its contents aren't present in the object image.
  virtual error_code isSectionVirtual(DataRefImpl Sec, bool &Res) const = 0;
  virtual error_code isSectionZeroInit(DataRefImpl Sec, bool &Res) const = 0;
  virtual error_code isSectionReadOnlyData(DataRefImpl Sec, bool &Res) const =0;
  virtual error_code sectionContainsSymbol(DataRefImpl Sec, DataRefImpl Symb,
                                           bool &Result) const = 0;
  virtual relocation_iterator getSectionRelBegin(DataRefImpl Sec) const = 0;
  virtual relocation_iterator getSectionRelEnd(DataRefImpl Sec) const = 0;


  // Same as above for RelocationRef.
  friend class RelocationRef;
  virtual error_code getRelocationNext(DataRefImpl Rel,
                                       RelocationRef &Res) const = 0;
  virtual error_code getRelocationAddress(DataRefImpl Rel,
                                          uint64_t &Res) const =0;
  virtual error_code getRelocationOffset(DataRefImpl Rel,
                                         uint64_t &Res) const =0;
  virtual error_code getRelocationSymbol(DataRefImpl Rel,
                                         SymbolRef &Res) const = 0;
  virtual error_code getRelocationType(DataRefImpl Rel,
                                       uint64_t &Res) const = 0;
  virtual error_code getRelocationTypeName(DataRefImpl Rel,
                                       SmallVectorImpl<char> &Result) const = 0;
  virtual error_code getRelocationAdditionalInfo(DataRefImpl Rel,
                                                 int64_t &Res) const = 0;
  virtual error_code getRelocationValueString(DataRefImpl Rel,
                                       SmallVectorImpl<char> &Result) const = 0;
  virtual error_code getRelocationHidden(DataRefImpl Rel, bool &Result) const {
    Result = false;
    return object_error::success;
  }

  // Same for LibraryRef
  friend class LibraryRef;
  virtual error_code getLibraryNext(DataRefImpl Lib, LibraryRef &Res) const = 0;
  virtual error_code getLibraryPath(DataRefImpl Lib, StringRef &Res) const = 0;

public:

  virtual symbol_iterator begin_symbols() const = 0;
  virtual symbol_iterator end_symbols() const = 0;

  virtual symbol_iterator begin_dynamic_symbols() const = 0;
  virtual symbol_iterator end_dynamic_symbols() const = 0;

  virtual section_iterator begin_sections() const = 0;
  virtual section_iterator end_sections() const = 0;

  virtual library_iterator begin_libraries_needed() const = 0;
  virtual library_iterator end_libraries_needed() const = 0;

  /// @brief The number of bytes used to represent an address in this object
  ///        file format.
  virtual uint8_t getBytesInAddress() const = 0;

  virtual StringRef getFileFormatName() const = 0;
  virtual /* Triple::ArchType */ unsigned getArch() const = 0;

  /// For shared objects, returns the name which this object should be
  /// loaded from at runtime. This corresponds to DT_SONAME on ELF and
  /// LC_ID_DYLIB (install name) on MachO.
  virtual StringRef getLoadName() const = 0;

  /// @returns Pointer to ObjectFile subclass to handle this type of object.
  /// @param ObjectPath The path to the object file. ObjectPath.isObject must
  ///        return true.
  /// @brief Create ObjectFile from path.
  static ObjectFile *createObjectFile(StringRef ObjectPath);
  static ObjectFile *createObjectFile(MemoryBuffer *Object);

  static inline bool classof(const Binary *v) {
    return v->isObject();
  }

public:
  static ObjectFile *createCOFFObjectFile(MemoryBuffer *Object);
  static ObjectFile *createELFObjectFile(MemoryBuffer *Object);
  static ObjectFile *createMachOObjectFile(MemoryBuffer *Object);
};

// Inline function definitions.
inline SymbolRef::SymbolRef(DataRefImpl SymbolP, const ObjectFile *Owner)
  : SymbolPimpl(SymbolP)
  , OwningObject(Owner) {}

inline bool SymbolRef::operator==(const SymbolRef &Other) const {
  return SymbolPimpl == Other.SymbolPimpl;
}

inline bool SymbolRef::operator<(const SymbolRef &Other) const {
  return SymbolPimpl < Other.SymbolPimpl;
}

inline error_code SymbolRef::getNext(SymbolRef &Result) const {
  return OwningObject->getSymbolNext(SymbolPimpl, Result);
}

inline error_code SymbolRef::getName(StringRef &Result) const {
  return OwningObject->getSymbolName(SymbolPimpl, Result);
}

inline error_code SymbolRef::getAddress(uint64_t &Result) const {
  return OwningObject->getSymbolAddress(SymbolPimpl, Result);
}

inline error_code SymbolRef::getFileOffset(uint64_t &Result) const {
  return OwningObject->getSymbolFileOffset(SymbolPimpl, Result);
}

inline error_code SymbolRef::getSize(uint64_t &Result) const {
  return OwningObject->getSymbolSize(SymbolPimpl, Result);
}

inline error_code SymbolRef::getNMTypeChar(char &Result) const {
  return OwningObject->getSymbolNMTypeChar(SymbolPimpl, Result);
}

inline error_code SymbolRef::getFlags(uint32_t &Result) const {
  return OwningObject->getSymbolFlags(SymbolPimpl, Result);
}

inline error_code SymbolRef::getSection(section_iterator &Result) const {
  return OwningObject->getSymbolSection(SymbolPimpl, Result);
}

inline error_code SymbolRef::getType(SymbolRef::Type &Result) const {
  return OwningObject->getSymbolType(SymbolPimpl, Result);
}

inline error_code SymbolRef::getValue(uint64_t &Val) const {
  return OwningObject->getSymbolValue(SymbolPimpl, Val);
}

inline DataRefImpl SymbolRef::getRawDataRefImpl() const {
  return SymbolPimpl;
}


/// SectionRef
inline SectionRef::SectionRef(DataRefImpl SectionP,
                              const ObjectFile *Owner)
  : SectionPimpl(SectionP)
  , OwningObject(Owner) {}

inline bool SectionRef::operator==(const SectionRef &Other) const {
  return SectionPimpl == Other.SectionPimpl;
}

inline bool SectionRef::operator<(const SectionRef &Other) const {
  return SectionPimpl < Other.SectionPimpl;
}

inline error_code SectionRef::getNext(SectionRef &Result) const {
  return OwningObject->getSectionNext(SectionPimpl, Result);
}

inline error_code SectionRef::getName(StringRef &Result) const {
  return OwningObject->getSectionName(SectionPimpl, Result);
}

inline error_code SectionRef::getAddress(uint64_t &Result) const {
  return OwningObject->getSectionAddress(SectionPimpl, Result);
}

inline error_code SectionRef::getSize(uint64_t &Result) const {
  return OwningObject->getSectionSize(SectionPimpl, Result);
}

inline error_code SectionRef::getContents(StringRef &Result) const {
  return OwningObject->getSectionContents(SectionPimpl, Result);
}

inline error_code SectionRef::getAlignment(uint64_t &Result) const {
  return OwningObject->getSectionAlignment(SectionPimpl, Result);
}

inline error_code SectionRef::isText(bool &Result) const {
  return OwningObject->isSectionText(SectionPimpl, Result);
}

inline error_code SectionRef::isData(bool &Result) const {
  return OwningObject->isSectionData(SectionPimpl, Result);
}

inline error_code SectionRef::isBSS(bool &Result) const {
  return OwningObject->isSectionBSS(SectionPimpl, Result);
}

inline error_code SectionRef::isRequiredForExecution(bool &Result) const {
  return OwningObject->isSectionRequiredForExecution(SectionPimpl, Result);
}

inline error_code SectionRef::isVirtual(bool &Result) const {
  return OwningObject->isSectionVirtual(SectionPimpl, Result);
}

inline error_code SectionRef::isZeroInit(bool &Result) const {
  return OwningObject->isSectionZeroInit(SectionPimpl, Result);
}

inline error_code SectionRef::isReadOnlyData(bool &Result) const {
  return OwningObject->isSectionReadOnlyData(SectionPimpl, Result);
}

inline error_code SectionRef::containsSymbol(SymbolRef S, bool &Result) const {
  return OwningObject->sectionContainsSymbol(SectionPimpl, S.SymbolPimpl,
                                             Result);
}

inline relocation_iterator SectionRef::begin_relocations() const {
  return OwningObject->getSectionRelBegin(SectionPimpl);
}

inline relocation_iterator SectionRef::end_relocations() const {
  return OwningObject->getSectionRelEnd(SectionPimpl);
}

inline DataRefImpl SectionRef::getRawDataRefImpl() const {
  return SectionPimpl;
}

/// RelocationRef
inline RelocationRef::RelocationRef(DataRefImpl RelocationP,
                              const ObjectFile *Owner)
  : RelocationPimpl(RelocationP)
  , OwningObject(Owner) {}

inline bool RelocationRef::operator==(const RelocationRef &Other) const {
  return RelocationPimpl == Other.RelocationPimpl;
}

inline error_code RelocationRef::getNext(RelocationRef &Result) const {
  return OwningObject->getRelocationNext(RelocationPimpl, Result);
}

inline error_code RelocationRef::getAddress(uint64_t &Result) const {
  return OwningObject->getRelocationAddress(RelocationPimpl, Result);
}

inline error_code RelocationRef::getOffset(uint64_t &Result) const {
  return OwningObject->getRelocationOffset(RelocationPimpl, Result);
}

inline error_code RelocationRef::getSymbol(SymbolRef &Result) const {
  return OwningObject->getRelocationSymbol(RelocationPimpl, Result);
}

inline error_code RelocationRef::getType(uint64_t &Result) const {
  return OwningObject->getRelocationType(RelocationPimpl, Result);
}

inline error_code RelocationRef::getTypeName(SmallVectorImpl<char> &Result)
  const {
  return OwningObject->getRelocationTypeName(RelocationPimpl, Result);
}

inline error_code RelocationRef::getAdditionalInfo(int64_t &Result) const {
  return OwningObject->getRelocationAdditionalInfo(RelocationPimpl, Result);
}

inline error_code RelocationRef::getValueString(SmallVectorImpl<char> &Result)
  const {
  return OwningObject->getRelocationValueString(RelocationPimpl, Result);
}

inline error_code RelocationRef::getHidden(bool &Result) const {
  return OwningObject->getRelocationHidden(RelocationPimpl, Result);
}

inline DataRefImpl RelocationRef::getRawDataRefImpl() const {
  return RelocationPimpl;
}

// Inline function definitions.
inline LibraryRef::LibraryRef(DataRefImpl LibraryP, const ObjectFile *Owner)
  : LibraryPimpl(LibraryP)
  , OwningObject(Owner) {}

inline bool LibraryRef::operator==(const LibraryRef &Other) const {
  return LibraryPimpl == Other.LibraryPimpl;
}

inline bool LibraryRef::operator<(const LibraryRef &Other) const {
  return LibraryPimpl < Other.LibraryPimpl;
}

inline error_code LibraryRef::getNext(LibraryRef &Result) const {
  return OwningObject->getLibraryNext(LibraryPimpl, Result);
}

inline error_code LibraryRef::getPath(StringRef &Result) const {
  return OwningObject->getLibraryPath(LibraryPimpl, Result);
}

} // end namespace object
} // end namespace llvm

#endif
