//===-- XCOFFDumper.cpp - XCOFF dumping utility -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an XCOFF specific dumper for llvm-readobj.
//
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "ObjDumper.h"
#include "llvm-readobj.h"
#include "llvm/Object/XCOFFObjectFile.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace llvm;
using namespace object;

namespace {

class XCOFFDumper : public ObjDumper {
public:
  XCOFFDumper(const XCOFFObjectFile &Obj, ScopedPrinter &Writer)
      : ObjDumper(Writer), Obj(Obj) {}

  void printFileHeaders() override;
  void printSectionHeaders() override;
  void printRelocations() override;
  void printSymbols() override;
  void printDynamicSymbols() override;
  void printUnwindInfo() override;
  void printStackMap() const override;
  void printNeededLibraries() override;

private:
  template <typename T> void printSectionHeaders(ArrayRef<T> Sections);

  const XCOFFObjectFile &Obj;

  // Least significant 3 bits are reserved.
  static constexpr unsigned SectionFlagsReservedMask = 0x7;
};
} // anonymous namespace

void XCOFFDumper::printFileHeaders() {
  DictScope DS(W, "FileHeader");
  W.printHex("Magic", Obj.getMagic());
  W.printNumber("NumberOfSections", Obj.getNumberOfSections());

  // Negative timestamp values are reserved for future use.
  int32_t TimeStamp = Obj.getTimeStamp();
  if (TimeStamp > 0) {
    // This handling of the time stamp assumes that the host system's time_t is
    // compatible with AIX time_t. If a platform is not compatible, the lit
    // tests will let us know.
    time_t TimeDate = TimeStamp;

    char FormattedTime[21] = {};
    size_t BytesWritten =
        strftime(FormattedTime, 21, "%Y-%m-%dT%H:%M:%SZ", gmtime(&TimeDate));
    if (BytesWritten)
      W.printHex("TimeStamp", FormattedTime, TimeStamp);
    else
      W.printHex("Timestamp", TimeStamp);
  } else {
    W.printHex("TimeStamp", TimeStamp == 0 ? "None" : "Reserved Value",
               TimeStamp);
  }

  // The number of symbol table entries is an unsigned value in 64-bit objects
  // and a signed value (with negative values being 'reserved') in 32-bit
  // objects.
  if (Obj.is64Bit()) {
    W.printHex("SymbolTableOffset", Obj.getSymbolTableOffset64());
    W.printNumber("SymbolTableEntries", Obj.getNumberOfSymbolTableEntries64());
  } else {
    W.printHex("SymbolTableOffset", Obj.getSymbolTableOffset32());
    int32_t SymTabEntries = Obj.getRawNumberOfSymbolTableEntries32();
    if (SymTabEntries >= 0)
      W.printNumber("SymbolTableEntries", SymTabEntries);
    else
      W.printHex("SymbolTableEntries", "Reserved Value", SymTabEntries);
  }

  W.printHex("OptionalHeaderSize", Obj.getOptionalHeaderSize());
  W.printHex("Flags", Obj.getFlags());

  // TODO FIXME Add support for the auxiliary header (if any) once
  // XCOFFObjectFile has the necessary support.
}

void XCOFFDumper::printSectionHeaders() {
  if (Obj.is64Bit())
    printSectionHeaders(Obj.sections64());
  else
    printSectionHeaders(Obj.sections32());
}

void XCOFFDumper::printRelocations() {
  llvm_unreachable("Unimplemented functionality for XCOFFDumper");
}

void XCOFFDumper::printSymbols() {
  llvm_unreachable("Unimplemented functionality for XCOFFDumper");
}

void XCOFFDumper::printDynamicSymbols() {
  llvm_unreachable("Unimplemented functionality for XCOFFDumper");
}

void XCOFFDumper::printUnwindInfo() {
  llvm_unreachable("Unimplemented functionality for XCOFFDumper");
}

void XCOFFDumper::printStackMap() const {
  llvm_unreachable("Unimplemented functionality for XCOFFDumper");
}

void XCOFFDumper::printNeededLibraries() {
  llvm_unreachable("Unimplemented functionality for XCOFFDumper");
}

static const EnumEntry<XCOFF::SectionTypeFlags> SectionTypeFlagsNames[] = {
#define ECase(X)                                                               \
  { #X, XCOFF::X }
    ECase(STYP_PAD),    ECase(STYP_DWARF), ECase(STYP_TEXT),
    ECase(STYP_DATA),   ECase(STYP_BSS),   ECase(STYP_EXCEPT),
    ECase(STYP_INFO),   ECase(STYP_TDATA), ECase(STYP_TBSS),
    ECase(STYP_LOADER), ECase(STYP_DEBUG), ECase(STYP_TYPCHK),
    ECase(STYP_OVRFLO)
#undef ECase
};

template <typename T>
void XCOFFDumper::printSectionHeaders(ArrayRef<T> Sections) {
  ListScope Group(W, "Sections");

  uint16_t Index = 1;
  for (const T &Sec : Sections) {
    DictScope SecDS(W, "Section");

    W.printNumber("Index", Index++);
    W.printString("Name", Sec.getName());

    W.printHex("PhysicalAddress", Sec.PhysicalAddress);
    W.printHex("VirtualAddress", Sec.VirtualAddress);
    W.printHex("Size", Sec.SectionSize);
    W.printHex("RawDataOffset", Sec.FileOffsetToRawData);
    W.printHex("RelocationPointer", Sec.FileOffsetToRelocationInfo);
    W.printHex("LineNumberPointer", Sec.FileOffsetToLineNumberInfo);

    // TODO Need to add overflow handling when NumberOfX == _OVERFLOW_MARKER
    // in 32-bit object files.
    W.printNumber("NumberOfRelocations", Sec.NumberOfRelocations);
    W.printNumber("NumberOfLineNumbers", Sec.NumberOfLineNumbers);

    // The most significant 16-bits represent the DWARF section subtype. For
    // now we just dump the section type flags.
    uint16_t Flags = Sec.Flags & 0xffffu;
    if (Flags & SectionFlagsReservedMask)
      W.printHex("Flags", "Reserved", Flags);
    else
      W.printEnum("Type", Flags, makeArrayRef(SectionTypeFlagsNames));
  }

  if (opts::SectionRelocations)
    report_fatal_error("Dumping section relocations is unimplemented");

  if (opts::SectionSymbols)
    report_fatal_error("Dumping symbols is unimplemented");

  if (opts::SectionData)
    report_fatal_error("Dumping section data is unimplemented");
}

namespace llvm {
std::error_code createXCOFFDumper(const object::ObjectFile *Obj,
                                  ScopedPrinter &Writer,
                                  std::unique_ptr<ObjDumper> &Result) {
  const XCOFFObjectFile *XObj = dyn_cast<XCOFFObjectFile>(Obj);
  if (!XObj)
    return readobj_error::unsupported_obj_file_format;

  Result.reset(new XCOFFDumper(*XObj, Writer));
  return readobj_error::success;
}
} // namespace llvm
