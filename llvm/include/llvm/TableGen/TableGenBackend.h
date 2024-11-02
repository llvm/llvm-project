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

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TableGen/Record.h"

namespace llvm {

class RecordKeeper;
class raw_ostream;

namespace TableGen::Emitter {
// Supports const and non-const forms of callback functions.
using FnT = function_ref<void(RecordKeeper &Records, raw_ostream &OS)>;

/// Creating an `Opt` object registers the command line option \p Name with
/// TableGen backend and associates the callback \p CB with that option. If
/// \p ByDefault is true, then that callback is applied by default if no
/// command line option was specified.
struct Opt {
  Opt(StringRef Name, FnT CB, StringRef Desc, bool ByDefault = false);
};

/// Convienence wrapper around `Opt` that registers `EmitterClass::run` as the
/// callback.
template <class EmitterC> class OptClass : Opt {
  static void run(RecordKeeper &RK, raw_ostream &OS) { EmitterC(RK).run(OS); }

public:
  OptClass(StringRef Name, StringRef Desc) : Opt(Name, run, Desc) {}
};

/// Apply callback for any command line option registered above. Returns false
/// is no callback was applied.
bool ApplyCallback(RecordKeeper &Records, raw_ostream &OS);

} // namespace TableGen::Emitter

/// emitSourceFileHeader - Output an LLVM style file header to the specified
/// raw_ostream.
void emitSourceFileHeader(StringRef Desc, raw_ostream &OS,
                          const RecordKeeper &Record = RecordKeeper());

} // namespace llvm

#endif // LLVM_TABLEGEN_TABLEGENBACKEND_H
