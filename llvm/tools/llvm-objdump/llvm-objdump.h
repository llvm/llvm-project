//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_OBJDUMP_LLVM_OBJDUMP_H
#define LLVM_TOOLS_LLVM_OBJDUMP_LLVM_OBJDUMP_H

#include "llvm/DebugInfo/DIContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Object/Archive.h"

namespace llvm {
class StringRef;

namespace object {
class COFFObjectFile;
class COFFImportFile;
class ELFObjectFileBase;
class ELFSectionRef;
class MachOObjectFile;
class MachOUniversalBinary;
class RelocationRef;
class XCOFFObjectFile;
}

extern cl::opt<bool> Demangle;

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

Error getELFRelocationValueString(const object::ELFObjectFileBase *Obj,
                                  const object::RelocationRef &Rel,
                                  llvm::SmallVectorImpl<char> &Result);
Error getCOFFRelocationValueString(const object::COFFObjectFile *Obj,
                                   const object::RelocationRef &Rel,
                                   llvm::SmallVectorImpl<char> &Result);
Error getWasmRelocationValueString(const object::WasmObjectFile *Obj,
                                   const object::RelocationRef &RelRef,
                                   llvm::SmallVectorImpl<char> &Result);
Error getMachORelocationValueString(const object::MachOObjectFile *Obj,
                                    const object::RelocationRef &RelRef,
                                    llvm::SmallVectorImpl<char> &Result);
Error getXCOFFRelocationValueString(const object::XCOFFObjectFile *Obj,
                                    const object::RelocationRef &RelRef,
                                    llvm::SmallVectorImpl<char> &Result);

uint64_t getELFSectionLMA(const object::ELFSectionRef& Sec);

bool isRelocAddressLess(object::RelocationRef A, object::RelocationRef B);
void parseInputMachO(StringRef Filename);
void parseInputMachO(object::MachOUniversalBinary *UB);
void printCOFFUnwindInfo(const object::COFFObjectFile *O);
void printMachOUnwindInfo(const object::MachOObjectFile *O);
void printMachOExportsTrie(const object::MachOObjectFile *O);
void printMachORebaseTable(object::MachOObjectFile *O);
void printMachOBindTable(object::MachOObjectFile *O);
void printMachOLazyBindTable(object::MachOObjectFile *O);
void printMachOWeakBindTable(object::MachOObjectFile *O);
void printELFFileHeader(const object::ObjectFile *O);
void printELFDynamicSection(const object::ObjectFile *Obj);
void printELFSymbolVersionInfo(const object::ObjectFile *Obj);
void printCOFFFileHeader(const object::ObjectFile *O);
void printCOFFSymbolTable(const object::COFFImportFile *I);
void printCOFFSymbolTable(const object::COFFObjectFile *O);
void printMachOFileHeader(const object::ObjectFile *O);
void printMachOLoadCommands(const object::ObjectFile *O);
void printWasmFileHeader(const object::ObjectFile *O);
void printExportsTrie(const object::ObjectFile *O);
void printRebaseTable(object::ObjectFile *O);
void printBindTable(object::ObjectFile *O);
void printLazyBindTable(object::ObjectFile *O);
void printWeakBindTable(object::ObjectFile *O);
void printRawClangAST(const object::ObjectFile *O);
void printRelocations(const object::ObjectFile *O);
void printDynamicRelocations(const object::ObjectFile *O);
void printSectionHeaders(const object::ObjectFile *O);
void printSectionContents(const object::ObjectFile *O);
void printSymbolTable(const object::ObjectFile *O, StringRef ArchiveName,
                      StringRef ArchitectureName = StringRef());
LLVM_ATTRIBUTE_NORETURN void reportError(StringRef File, Twine Message);
LLVM_ATTRIBUTE_NORETURN void reportError(Error E, StringRef FileName,
                                         StringRef ArchiveName = "",
                                         StringRef ArchitectureName = "");
void reportWarning(Twine Message, StringRef File);

template <typename T, typename... Ts>
T unwrapOrError(Expected<T> EO, Ts &&... Args) {
  if (EO)
    return std::move(*EO);
  reportError(EO.takeError(), std::forward<Ts>(Args)...);
}

std::string getFileNameForError(const object::Archive::Child &C,
                                unsigned Index);

} // end namespace llvm

#endif
