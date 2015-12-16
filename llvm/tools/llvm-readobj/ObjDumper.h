//===-- ObjDumper.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_READOBJ_OBJDUMPER_H
#define LLVM_TOOLS_LLVM_READOBJ_OBJDUMPER_H

#include <memory>
#include <system_error>

namespace llvm {
namespace object {
class COFFImportFile;
class ObjectFile;
}

class StreamWriter;

class ObjDumper {
public:
  ObjDumper(StreamWriter& Writer);
  virtual ~ObjDumper();

  virtual void printFileHeaders() = 0;
  virtual void printSections() = 0;
  virtual void printRelocations() = 0;
  virtual void printSymbols() = 0;
  virtual void printDynamicSymbols() = 0;
  virtual void printUnwindInfo() = 0;

  // Only implemented for ELF at this time.
  virtual void printDynamicRelocations() { }
  virtual void printDynamicTable() { }
  virtual void printNeededLibraries() { }
  virtual void printProgramHeaders() { }
  virtual void printHashTable() { }
  virtual void printGnuHashTable() { }
  virtual void printLoadName() {}
  virtual void printVersionInfo() {}

  // Only implemented for ARM ELF at this time.
  virtual void printAttributes() { }

  // Only implemented for MIPS ELF at this time.
  virtual void printMipsPLTGOT() { }
  virtual void printMipsABIFlags() { }
  virtual void printMipsReginfo() { }

  // Only implemented for PE/COFF.
  virtual void printCOFFImports() { }
  virtual void printCOFFExports() { }
  virtual void printCOFFDirectives() { }
  virtual void printCOFFBaseReloc() { }
  virtual void printCodeViewDebugInfo() { }

  // Only implemented for MachO.
  virtual void printMachODataInCode() { }
  virtual void printMachOVersionMin() { }
  virtual void printMachODysymtab() { }
  virtual void printMachOSegment() { }
  virtual void printMachOIndirectSymbols() { }
  virtual void printMachOLinkerOptions() { }

  virtual void printStackMap() const = 0;

protected:
  StreamWriter& W;
};

std::error_code createCOFFDumper(const object::ObjectFile *Obj,
                                 StreamWriter &Writer,
                                 std::unique_ptr<ObjDumper> &Result);

std::error_code createELFDumper(const object::ObjectFile *Obj,
                                StreamWriter &Writer,
                                std::unique_ptr<ObjDumper> &Result);

std::error_code createMachODumper(const object::ObjectFile *Obj,
                                  StreamWriter &Writer,
                                  std::unique_ptr<ObjDumper> &Result);

void dumpCOFFImportFile(const object::COFFImportFile *File);

} // namespace llvm

#endif
