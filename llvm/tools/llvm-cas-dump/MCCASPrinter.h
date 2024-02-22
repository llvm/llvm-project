//===-----------------------------------------------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_CAS_DUMP_MCCASPRINTER_H
#define LLVM_TOOLS_LLVM_CAS_DUMP_MCCASPRINTER_H

#include "CASDWARFObject.h"
#include "llvm/MCCAS/MCCASDebugV1.h"
#include "llvm/MCCAS/MCCASObjectV1.h"
#include "llvm/Support/Error.h"

namespace llvm {
class raw_ostream;
class DWARFContext;

namespace mccasformats {
namespace v1 {

struct PrinterOptions {
  bool DwarfSectionsOnly = false;
  bool DwarfDump = false;
  bool HexDump = false;
  bool HexDumpOneLine = false;
  bool Verbose = false;
  bool DIERefs = false;
};

struct MCCASPrinter {
  /// Creates a printer object capable of printing MCCAS objects inside `CAS`.
  /// Output is sent to `OS`.
  MCCASPrinter(PrinterOptions Options, cas::ObjectStore &CAS, raw_ostream &OS);
  ~MCCASPrinter();

  /// If `CASObj` is an MCObject, prints its contents and all nodes referenced
  /// by it recursively. If CASObj or any of its children are not MCObjects, an
  /// error is returned.
  Error printMCObject(cas::ObjectRef CASObj, CASDWARFObject &Obj,
                      DWARFContext *DWARFCtx = nullptr);

  /// Prints the contents of `MCObject` and all nodes referenced by it
  /// recursively. If any of its children are not MCObjects, an error is
  /// returned.
  Error printMCObject(MCObjectProxy MCObj, CASDWARFObject &Obj,
                      DWARFContext *DWARFCtx);

  Expected<CASDWARFObject> discoverDwarfSections(cas::ObjectRef CASObj);

private:
  PrinterOptions Options;
  llvm::mccasformats::v1::MCSchema MCSchema;
  int Indent;
  raw_ostream &OS;

  Error printSimpleNested(MCObjectProxy AssemblerRef, CASDWARFObject &Obj,
                          DWARFContext *DWARFCtx);
};
} // namespace v1
} // namespace mccasformats
} // namespace llvm
#endif
