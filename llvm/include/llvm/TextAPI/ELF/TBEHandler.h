//===- TBEHandler.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "llvm/Support/Error.h"
#include <memory>

namespace llvm {

class raw_ostream;
class Error;
class StringRef;

namespace elfabi {

class ELFStub;

const VersionTuple TBEVersionCurrent(1, 0);

/// Attempts to read an ELF interface file from a StringRef buffer.
Expected<std::unique_ptr<ELFStub>> readTBEFromBuffer(StringRef Buf);

/// Attempts to write an ELF interface file to a raw_ostream.
Error writeTBEToOutputStream(raw_ostream &OS, const ELFStub &Stub);

} // end namespace elfabi
} // end namespace llvm

#endif // LLVM_TEXTAPI_ELF_TBEHANDLER_H
