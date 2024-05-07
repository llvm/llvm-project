//===- DXIL.h - Abstractions for DXIL constructs ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file This file defines various abstractions for transforming between DXIL's
// and LLVM's representations of shader metadata.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_DXIL_H
#define LLVM_TRANSFORMS_UTILS_DXIL_H

#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
class Module;

namespace dxil {

class DXILVersion {
  unsigned Major = 0;
  unsigned Minor = 0;

public:
  DXILVersion() = default;
  DXILVersion(unsigned Major, unsigned Minor) : Major(Major), Minor(Minor) {}

  /// Get the DXILVersion for \c M
  static Expected<DXILVersion> get(Module &M);
  /// Read the DXILVersion from the DXIL metadata in \c M
  static Expected<DXILVersion> readDXIL(Module &M);

  /// Returns true if no DXILVersion is set
  bool empty() { return Major == 0 && Minor == 0; }

  /// Remove any non-DXIL LLVM representations of the DXILVersion from \c M.
  void strip(Module &M);
  /// Embed the LLVM representation of the DXILVersion into \c M.
  void embed(Module &M);
  /// Remove any DXIL representation of the DXILVersion from \c M.
  void stripDXIL(Module &M);
  /// Embed a DXIL representation of the DXILVersion into \c M.
  void embedDXIL(Module &M);

  void print(raw_ostream &OS) const {
    // Format like Triple ArchName.
    OS << "dxilv" << Major << "." << Minor;
  }
  LLVM_DUMP_METHOD void dump() const { print(errs()); }
};

inline raw_ostream &operator<<(raw_ostream &OS, const DXILVersion &V) {
  V.print(OS);
  return OS;
}

} // namespace dxil
} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_DXIL_H

