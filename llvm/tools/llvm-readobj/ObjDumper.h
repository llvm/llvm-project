//===-- ObjDumper.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_READOBJ_OBJDUMPER_H
#define LLVM_TOOLS_LLVM_READOBJ_OBJDUMPER_H

#include <memory>
#include <system_error>

#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"

#include <unordered_set>

namespace llvm {
namespace object {
class COFFImportFile;
class ObjectFile;
class XCOFFObjectFile;
class ELFObjectFileBase;
}
namespace codeview {
class GlobalTypeTableBuilder;
class MergingTypeTableBuilder;
} // namespace codeview

class ScopedPrinter;

class ObjDumper {
public:
  ObjDumper(ScopedPrinter &Writer, StringRef ObjName);
  virtual ~ObjDumper();

  virtual bool canDumpContent() { return true; }

  virtual void printFileHeaders() = 0;
  virtual void printSectionHeaders() = 0;
  virtual void printRelocations() = 0;
  virtual void printSymbols(bool PrintSymbols, bool PrintDynamicSymbols) {
    if (PrintSymbols)
      printSymbols();
    if (PrintDynamicSymbols)
      printDynamicSymbols();
  }
  virtual void printProgramHeaders(bool PrintProgramHeaders,
                                   cl::boolOrDefault PrintSectionMapping) {
    if (PrintProgramHeaders)
      printProgramHeaders();
    if (PrintSectionMapping == cl::BOU_TRUE)
      printSectionMapping();
  }

  virtual void printUnwindInfo() = 0;

  // Only implemented for ELF at this time.
  virtual void printDependentLibs() {}
  virtual void printDynamicRelocations() { }
  virtual void printDynamicTable() { }
  virtual void printNeededLibraries() { }
  virtual void printSectionAsHex(StringRef SectionName) {}
  virtual void printHashTable() { }
  virtual void printGnuHashTable() {}
  virtual void printHashSymbols() {}
  virtual void printLoadName() {}
  virtual void printVersionInfo() {}
  virtual void printGroupSections() {}
  virtual void printHashHistograms() {}
  virtual void printCGProfile() {}
  virtual void printBBAddrMaps() {}
  virtual void printAddrsig() {}
  virtual void printNotes() {}
  virtual void printELFLinkerOptions() {}
  virtual void printStackSizes() {}
  virtual void printSectionDetails() {}
  virtual void printArchSpecificInfo() {}

  // Only implemented for PE/COFF.
  virtual void printCOFFImports() { }
  virtual void printCOFFExports() { }
  virtual void printCOFFDirectives() { }
  virtual void printCOFFBaseReloc() { }
  virtual void printCOFFDebugDirectory() { }
  virtual void printCOFFTLSDirectory() {}
  virtual void printCOFFResources() {}
  virtual void printCOFFLoadConfig() { }
  virtual void printCodeViewDebugInfo() { }
  virtual void
  mergeCodeViewTypes(llvm::codeview::MergingTypeTableBuilder &CVIDs,
                     llvm::codeview::MergingTypeTableBuilder &CVTypes,
                     llvm::codeview::GlobalTypeTableBuilder &GlobalCVIDs,
                     llvm::codeview::GlobalTypeTableBuilder &GlobalCVTypes,
                     bool GHash) {}

  // Only implement for XCOFF
  virtual void printAuxiliaryHeader() {}

  // Only implemented for MachO.
  virtual void printMachODataInCode() { }
  virtual void printMachOVersionMin() { }
  virtual void printMachODysymtab() { }
  virtual void printMachOSegment() { }
  virtual void printMachOIndirectSymbols() { }
  virtual void printMachOLinkerOptions() { }

  // Currently only implemented for XCOFF.
  virtual void printStringTable() { }

  virtual void printStackMap() const = 0;

  void printAsStringList(StringRef StringContent, size_t StringDataOffset = 0);

  void printSectionsAsString(const object::ObjectFile &Obj,
                             ArrayRef<std::string> Sections);
  void printSectionsAsHex(const object::ObjectFile &Obj,
                          ArrayRef<std::string> Sections);

  std::function<Error(const Twine &Msg)> WarningHandler;
  void reportUniqueWarning(Error Err) const;
  void reportUniqueWarning(const Twine &Msg) const;

protected:
  ScopedPrinter &W;

private:
  virtual void printSymbols() {}
  virtual void printDynamicSymbols() {}
  virtual void printProgramHeaders() {}
  virtual void printSectionMapping() {}

  std::unordered_set<std::string> Warnings;
};

std::unique_ptr<ObjDumper> createCOFFDumper(const object::COFFObjectFile &Obj,
                                            ScopedPrinter &Writer);

std::unique_ptr<ObjDumper> createELFDumper(const object::ELFObjectFileBase &Obj,
                                           ScopedPrinter &Writer);

std::unique_ptr<ObjDumper> createMachODumper(const object::MachOObjectFile &Obj,
                                             ScopedPrinter &Writer);

std::unique_ptr<ObjDumper> createWasmDumper(const object::WasmObjectFile &Obj,
                                            ScopedPrinter &Writer);

std::unique_ptr<ObjDumper> createXCOFFDumper(const object::XCOFFObjectFile &Obj,
                                             ScopedPrinter &Writer);

void dumpCOFFImportFile(const object::COFFImportFile *File,
                        ScopedPrinter &Writer);

void dumpCodeViewMergedTypes(ScopedPrinter &Writer,
                             ArrayRef<ArrayRef<uint8_t>> IpiRecords,
                             ArrayRef<ArrayRef<uint8_t>> TpiRecords);

} // namespace llvm

#endif
