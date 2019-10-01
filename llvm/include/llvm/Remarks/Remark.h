//===-- llvm/Remarks/Remark.h - The remark type -----------------*- C++/-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an abstraction for handling remarks.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_REMARKS_REMARK_H
#define LLVM_REMARKS_REMARK_H

#include "llvm-c/Remarks.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CBindingWrapping.h"
#include <string>

namespace llvm {
namespace remarks {

constexpr uint64_t Version = 0;
constexpr StringRef Magic("REMARKS", 7);

/// The debug location used to track a remark back to the source file.
struct RemarkLocation {
  /// Absolute path of the source file corresponding to this remark.
  StringRef SourceFilePath;
  unsigned SourceLine;
  unsigned SourceColumn;
};

// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(RemarkLocation, LLVMRemarkDebugLocRef)

/// A key-value pair with a debug location that is used to display the remarks
/// at the right place in the source.
struct Argument {
  StringRef Key;
  // FIXME: We might want to be able to store other types than strings here.
  StringRef Val;
  // If set, the debug location corresponding to the value.
  Optional<RemarkLocation> Loc;
};

// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(Argument, LLVMRemarkArgRef)

/// The type of the remark.
enum class Type {
  Unknown,
  Passed,
  Missed,
  Analysis,
  AnalysisFPCommute,
  AnalysisAliasing,
  Failure,
  LastTypeValue = Failure
};

/// A remark type used for both emission and parsing.
struct Remark {
  /// The type of the remark.
  Type RemarkType = Type::Unknown;

  /// Name of the pass that triggers the emission of this remark.
  StringRef PassName;

  /// Textual identifier for the remark (single-word, camel-case). Can be used
  /// by external tools reading the output file for remarks to identify the
  /// remark.
  StringRef RemarkName;

  /// Mangled name of the function that triggers the emssion of this remark.
  StringRef FunctionName;

  /// The location in the source file of the remark.
  Optional<RemarkLocation> Loc;

  /// If profile information is available, this is the number of times the
  /// corresponding code was executed in a profile instrumentation run.
  Optional<uint64_t> Hotness;

  /// Arguments collected via the streaming interface.
  ArrayRef<Argument> Args;

  /// Return a message composed from the arguments as a string.
  std::string getArgsAsMsg() const;
};

// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(Remark, LLVMRemarkEntryRef)

} // end namespace remarks
} // end namespace llvm

#endif /* LLVM_REMARKS_REMARK_H */
