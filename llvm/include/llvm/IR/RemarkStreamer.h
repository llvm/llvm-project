//===- llvm/IR/RemarkStreamer.h - Remark Streamer ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the main interface for outputting remarks.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_REMARKSTREAMER_H
#define LLVM_IR_REMARKSTREAMER_H

#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Remarks/RemarkSerializer.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

namespace llvm {
/// Streamer for remarks.
class RemarkStreamer {
  /// The filename that the remark diagnostics are emitted to.
  const std::string Filename;
  /// The regex used to filter remarks based on the passes that emit them.
  Optional<Regex> PassFilter;
  /// The object used to serialize the remarks to a specific format.
  std::unique_ptr<remarks::Serializer> Serializer;

  /// Temporary buffer for converting diagnostics into remark objects. This is
  /// used for the remark arguments that are converted from a vector of
  /// diagnostic arguments to a vector of remark arguments.
  SmallVector<remarks::Argument, 8> TmpArgs;
  /// Convert diagnostics into remark objects. The result uses \p TmpArgs as a
  /// temporary buffer for the remark arguments, and relies on all the strings
  /// to be kept in memory until the next call to `toRemark`.
  /// The lifetime of the members of the result is bound to the lifetime of both
  /// the remark streamer and the LLVM diagnostics.
  remarks::Remark toRemark(const DiagnosticInfoOptimizationBase &Diag);

public:
  RemarkStreamer(StringRef Filename,
                 std::unique_ptr<remarks::Serializer> Serializer);
  /// Return the filename that the remark diagnostics are emitted to.
  StringRef getFilename() const { return Filename; }
  /// Return stream that the remark diagnostics are emitted to.
  raw_ostream &getStream() { return Serializer->OS; }
  /// Return the serializer used for this stream.
  remarks::Serializer &getSerializer() { return *Serializer; }
  /// Set a pass filter based on a regex \p Filter.
  /// Returns an error if the regex is invalid.
  Error setFilter(StringRef Filter);
  /// Emit a diagnostic through the streamer.
  void emit(const DiagnosticInfoOptimizationBase &Diag);
};
} // end namespace llvm

#endif // LLVM_IR_REMARKSTREAMER_H
