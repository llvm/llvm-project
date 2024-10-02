//===- TableGenBackend.cpp - Utilities for TableGen Backends ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides useful services for TableGen backends...
//
//===----------------------------------------------------------------------===//

#include "llvm/TableGen/TableGenBackend.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstddef>

using namespace llvm;
using namespace TableGen::Emitter;

const size_t MAX_LINE_LEN = 80U;

// CommandLine options of class type are not directly supported with some
// specific exceptions like std::string which are safe to copy. In our case,
// the `FnT` function_ref object is also safe to copy. So provide a
// specialization of `OptionValue` for `FnT` type that stores it as a copy.
// This is essentially similar to OptionValue<std::string> specialization for
// strings.
template <> struct cl::OptionValue<FnT> final : cl::OptionValueCopy<FnT> {
  OptionValue() = default;

  OptionValue(const FnT &V) { this->setValue(V); }

  OptionValue<FnT> &operator=(const FnT &V) {
    setValue(V);
    return *this;
  }
};

namespace {
struct OptCreatorT {
  static void *call() {
    return new cl::opt<FnT>(cl::desc("Action to perform:"));
  }
};
} // namespace

static ManagedStatic<cl::opt<FnT>, OptCreatorT> CallbackFunction;

Opt::Opt(StringRef Name, FnT CB, StringRef Desc, bool ByDefault) {
  if (ByDefault)
    CallbackFunction->setInitialValue(CB);
  CallbackFunction->getParser().addLiteralOption(Name, CB, Desc);
}

/// Apply callback specified on the command line. Returns true if no callback
/// was applied.
bool llvm::TableGen::Emitter::ApplyCallback(RecordKeeper &Records,
                                            raw_ostream &OS) {
  FnT Fn = CallbackFunction->getValue();
  if (!Fn)
    return true;
  Fn(Records, OS);
  return false;
}

static void printLine(raw_ostream &OS, const Twine &Prefix, char Fill,
                      StringRef Suffix) {
  size_t Pos = (size_t)OS.tell();
  assert((Prefix.str().size() + Suffix.size() <= MAX_LINE_LEN) &&
         "header line exceeds max limit");
  OS << Prefix;
  for (size_t i = (size_t)OS.tell() - Pos, e = MAX_LINE_LEN - Suffix.size();
         i < e; ++i)
    OS << Fill;
  OS << Suffix << '\n';
}

void llvm::emitSourceFileHeader(StringRef Desc, raw_ostream &OS,
                                const RecordKeeper &Record) {
  printLine(OS, "/*===- TableGen'erated file ", '-', "*- C++ -*-===*\\");
  StringRef Prefix("|* ");
  StringRef Suffix(" *|");
  printLine(OS, Prefix, ' ', Suffix);
  size_t PSLen = Prefix.size() + Suffix.size();
  assert(PSLen < MAX_LINE_LEN);
  size_t Pos = 0U;
  do {
    size_t Length = std::min(Desc.size() - Pos, MAX_LINE_LEN - PSLen);
    printLine(OS, Prefix + Desc.substr(Pos, Length), ' ', Suffix);
    Pos += Length;
  } while (Pos < Desc.size());
  printLine(OS, Prefix, ' ', Suffix);
  printLine(OS, Prefix + "Automatically generated file, do not edit!", ' ',
            Suffix);

  // Print the filename of source file.
  if (!Record.getInputFilename().empty())
    printLine(
        OS, Prefix + "From: " + sys::path::filename(Record.getInputFilename()),
        ' ', Suffix);
  printLine(OS, Prefix, ' ', Suffix);
  printLine(OS, "\\*===", '-', "===*/");
  OS << '\n';
}
