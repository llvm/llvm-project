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
#include "llvm/Remarks/RemarkStringTable.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

namespace llvm {
/// Streamer for remarks.
class RemarkStreamer {
  /// The filename that the remark diagnostics are emitted to.
  const std::string Filename;
  /// The open raw_ostream that the remark diagnostics are emitted to.
  raw_ostream &OS;
  /// The regex used to filter remarks based on the passes that emit them.
  Optional<Regex> PassFilter;

  /// The YAML streamer.
  yaml::Output YAMLOutput;

  /// The string table containing all the unique strings used in the output.
  /// The table will be serialized in a section to be consumed after the
  /// compilation.
  remarks::StringTable StrTab;

public:
  RemarkStreamer(StringRef Filename, raw_ostream& OS);
  /// Return the filename that the remark diagnostics are emitted to.
  StringRef getFilename() const { return Filename; }
  /// Return stream that the remark diagnostics are emitted to.
  raw_ostream &getStream() { return OS; }
  /// Set a pass filter based on a regex \p Filter.
  /// Returns an error if the regex is invalid.
  Error setFilter(StringRef Filter);
  /// Emit a diagnostic through the streamer.
  void emit(const DiagnosticInfoOptimizationBase &Diag);
  /// The string table used during emission.
  remarks::StringTable &getStringTable() { return StrTab; }
  const remarks::StringTable &getStringTable() const { return StrTab; }
};
} // end namespace llvm

#endif // LLVM_IR_REMARKSTREAMER_H
