//===-- GCMetadataPrinter.cpp - Garbage collection infrastructure ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the abstract base class GCMetadataPrinter.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GCMetadataPrinter.h"

using namespace llvm;

GCMetadataPrinter::GCMetadataPrinter() { }

GCMetadataPrinter::~GCMetadataPrinter() { }

void GCMetadataPrinter::beginAssembly(raw_ostream &OS, AsmPrinter &AP,
                                      const TargetAsmInfo &TAI) {
  // Default is no action.
}

void GCMetadataPrinter::finishAssembly(raw_ostream &OS, AsmPrinter &AP,
                                       const TargetAsmInfo &TAI) {
  // Default is no action.
}
