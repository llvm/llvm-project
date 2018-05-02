//===-- ObjDumper.cpp - Base dumper class -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements ObjDumper.
///
//===----------------------------------------------------------------------===//

#include "ObjDumper.h"
#include "Error.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

ObjDumper::ObjDumper(ScopedPrinter &Writer) : W(Writer) {}

ObjDumper::~ObjDumper() {
}

} // namespace llvm
