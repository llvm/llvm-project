//===--- llvm-objdump.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_OBJDUMP_LLVM_OBJDUMP_H
#define LLVM_TOOLS_LLVM_OBJDUMP_LLVM_OBJDUMP_H

#include "llvm/ADT/StringSet.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
class StringRef;
class Twine;

namespace object {
class ELFObjectFileBase;
class ELFSectionRef;
class MachOObjectFile;
class MachOUniversalBinary;
class RelocationRef;
struct VersionEntry;
} // namespace object

namespace objdump {

enum DebugVarsFormat { DVDisabled, DVUnicode, DVASCII, DVInvalid };

extern bool ArchiveHeaders;
extern int DbgIndent;
extern DebugVarsFormat DbgVariables;
extern bool Demangle;
extern bool Disassemble;
extern bool DisassembleAll;
extern DIDumpType DwarfDumpType;
extern std::vector<std::string> FilterSections;
extern bool LeadingAddr;
extern std::vector<std::string> MAttrs;
extern std::string MCPU;
extern std::string Prefix;
extern uint32_t PrefixStrip;
extern bool PrintImmHex;
extern bool PrintLines;
extern bool PrintSource;
extern bool PrivateHeaders;
extern bool Relocations;
extern bool SectionHeaders;
extern bool SectionContents;
extern bool ShowRawInsn;
extern bool SymbolDescription;
extern bool SymbolTable;
extern std::string TripleName;
extern bool UnwindInfo;

extern StringSet<> FoundSectionSet;

typedef std::function<bool(llvm::object::SectionRef const &)> FilterPredicate;

/// A filtered iterator for SectionRefs that skips sections based on some given
/// predicate.
class SectionFilterIterator {
public:
  SectionFilterIterator(FilterPredicate P,
                        llvm::object::section_iterator const &I,
                        llvm::object::section_iterator const &E)
      : Predicate(std::move(P)), Iterator(I), End(E) {
    ScanPredicate();
  }
  const llvm::object::SectionRef &operator*() const { return *Iterator; }
  SectionFilterIterator &operator++() {
    ++Iterator;
    ScanPredicate();
    return *this;
  }
  bool operator!=(SectionFilterIterator const &Other) const {
    return Iterator != Other.Iterator;
  }

private:
  void ScanPredicate() {
    while (Iterator != End && !Predicate(*Iterator)) {
      ++Iterator;
    }
  }
  FilterPredicate Predicate;
  llvm::object::section_iterator Iterator;
  llvm::object::section_iterator End;
};

/// Creates an iterator range of SectionFilterIterators for a given Object and
/// predicate.
class SectionFilter {
public:
  SectionFilter(FilterPredicate P, llvm::object::ObjectFile const &O)
      : Predicate(std::move(P)), Object(O) {}
  SectionFilterIterator begin() {
    return SectionFilterIterator(Predicate, Object.section_begin(),
                                 Object.section_end());
  }
  SectionFilterIterator end() {
    return SectionFilterIterator(Predicate, Object.section_end(),
                                 Object.section_end());
  }

private:
  FilterPredicate Predicate;
  llvm::object::ObjectFile const &Object;
};

// Various helper functions.

/// Creates a SectionFilter with a standard predicate that conditionally skips
/// sections when the --section objdump flag is provided.
///
/// Idx is an optional output parameter that keeps track of which section index
/// this is. This may be different than the actual section number, as some
/// sections may be filtered (e.g. symbol tables).
SectionFilter ToolSectionFilter(llvm::object::ObjectFile const &O,
                                uint64_t *Idx = nullptr);

bool isRelocAddressLess(object::RelocationRef A, object::RelocationRef B);
void printRelocations(const object::ObjectFile *O);
void printDynamicRelocations(const object::ObjectFile *O);
void printSectionHeaders(const object::ObjectFile *O);
void printSectionContents(const object::ObjectFile *O);
void printSymbolTable(const object::ObjectFile *O, StringRef ArchiveName,
                      StringRef ArchitectureName = StringRef(),
                      bool DumpDynamic = false);
void printSymbol(const object::ObjectFile *O, const object::SymbolRef &Symbol,
                 ArrayRef<object::VersionEntry> SymbolVersions,
                 StringRef FileName, StringRef ArchiveName,
                 StringRef ArchitectureName, bool DumpDynamic);
[[noreturn]] void reportError(StringRef File, const Twine &Message);
[[noreturn]] void reportError(Error E, StringRef FileName,
                              StringRef ArchiveName = "",
                              StringRef ArchitectureName = "");
void reportWarning(const Twine &Message, StringRef File);

template <typename T, typename... Ts>
T unwrapOrError(Expected<T> EO, Ts &&... Args) {
  if (EO)
    return std::move(*EO);
  reportError(EO.takeError(), std::forward<Ts>(Args)...);
}

std::string getFileNameForError(const object::Archive::Child &C,
                                unsigned Index);
SymbolInfoTy createSymbolInfo(const object::ObjectFile *Obj,
                              const object::SymbolRef &Symbol);

} // namespace objdump
} // end namespace llvm

#endif
