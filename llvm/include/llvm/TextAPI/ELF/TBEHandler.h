//===- TBEHandler.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-----------------------------------------------------------------------===/
///
/// \file
/// This file declares an interface for reading and writing .tbe (text-based
/// ELF) files.
///
//===-----------------------------------------------------------------------===/

#ifndef LLVM_TEXTAPI_ELF_TBEHANDLER_H
#define LLVM_TEXTAPI_ELF_TBEHANDLER_H

#include "llvm/Support/VersionTuple.h"
#include <memory>

namespace llvm {

class raw_ostream;
class Error;
class StringRef;
class VersionTuple;

namespace elfabi {

class ELFStub;

const VersionTuple TBEVersionCurrent(1, 0);

class TBEHandler {
public:
  /// Attempts to read an ELF interface file from a StringRef buffer.
  std::unique_ptr<ELFStub> readFile(StringRef Buf);

  /// Attempts to write an ELF interface file to a raw_ostream.
  Error writeFile(raw_ostream &OS, const ELFStub &Stub);
};
} // end namespace elfabi
} // end namespace llvm

#endif // LLVM_TEXTAPI_ELF_TBEHANDLER_H
