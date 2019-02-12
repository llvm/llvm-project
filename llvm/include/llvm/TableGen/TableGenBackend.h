//===- llvm/TableGen/TableGenBackend.h - Backend utilities ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Useful utilities for TableGen backends.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TABLEGEN_TABLEGENBACKEND_H
#define LLVM_TABLEGEN_TABLEGENBACKEND_H

namespace llvm {

class StringRef;
class raw_ostream;

/// emitSourceFileHeader - Output an LLVM style file header to the specified
/// raw_ostream.
void emitSourceFileHeader(StringRef Desc, raw_ostream &OS);

extern bool TimeRegions;

} // End llvm namespace

#endif
