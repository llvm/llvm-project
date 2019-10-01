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
  const XCOFFObjectFile &Obj;
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

  W.printHex("SymbolTableOffset", Obj.getSymbolTableOffset());
  int32_t SymTabEntries = Obj.getRawNumberOfSymbolTableEntries();
  if (SymTabEntries >= 0)
    W.printNumber("SymbolTableEntries", SymTabEntries);
  else
    W.printHex("SymbolTableEntries", "Reserved Value", SymTabEntries);

  W.printHex("OptionalHeaderSize", Obj.getOptionalHeaderSize());
  W.printHex("Flags", Obj.getFlags());

  // TODO FIXME Add support for the auxiliary header (if any) once
  // XCOFFObjectFile has the necessary support.
}

void XCOFFDumper::printSectionHeaders() {
  llvm_unreachable("Unimplemented functionality for XCOFFDumper");
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
