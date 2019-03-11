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

  /// The YAML streamer.
  yaml::Output YAMLOutput;

public:
  RemarkStreamer(StringRef Filename, raw_ostream& OS);
  /// Return the filename that the remark diagnostics are emitted to.
  StringRef getFilename() const { return Filename; }
  /// Return stream that the remark diagnostics are emitted to.
  raw_ostream &getStream() { return OS; }
  /// Emit a diagnostic through the streamer.
  void emit(const DiagnosticInfoOptimizationBase &Diag);
};
} // end namespace llvm

#endif // LLVM_IR_REMARKSTREAMER_H
