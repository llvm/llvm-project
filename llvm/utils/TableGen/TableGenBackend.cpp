//===- TableGenBackend.cpp - Base class for TableGen Backends ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides useful services for TableGen backends...
//
//===----------------------------------------------------------------------===//

#include "TableGenBackend.h"
#include "Record.h"
using namespace llvm;

void TableGenBackend::EmitSourceFileHeader(const std::string &Desc,
                                           std::ostream &OS) const {
  OS << "//===- TableGen'erated file -------------------------------------*-"
       " C++ -*-===//\n//\n// " << Desc << "\n//\n// Automatically generate"
       "d file, do not edit!\n//\n//===------------------------------------"
       "----------------------------------===//\n\n";
}

