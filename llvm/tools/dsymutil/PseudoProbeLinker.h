//===- tools/dsymutil/PseudoProbeLinker.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_DSYMUTIL_PSEUDOPROBELINKER_H
#define LLVM_TOOLS_DSYMUTIL_PSEUDOPROBELINKER_H

#include "LinkUtils.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include <string>

namespace llvm {
class Triple;
namespace dsymutil {

class BinaryHolder;
class DebugMap;

/// Collect and merge the __probes and __probe_descs sections from the debug map
/// object into the companion Contents/Resources/Profiling/pseudo_probes-<arch>.
/// Note: the probe sections contain no relocations.
class PseudoProbeLinker {
  BinaryHolder &BinHolder;
  const LinkOptions &Options;
  std::string Probes;
  std::string ProbeDescs;

  Error emit(const Triple &TheTriple) const;

public:
  PseudoProbeLinker(BinaryHolder &BinHolder, const LinkOptions &Options)
      : BinHolder(BinHolder), Options(Options) {}

  /// Emit the collected probe sections as a single MachO object sidecar for the
  /// \p TheTriple architecture. No-op if nothing was collected.
  bool link(const DebugMap &Map);

  /// Append the pseudo-probe sections found in \p Obj, if any.
  void collect(const object::ObjectFile &Obj);

  bool empty() const { return Probes.empty() && ProbeDescs.empty(); }
  StringRef getProbes() const { return Probes; }
  StringRef getProbeDescs() const { return ProbeDescs; }
};

} // end namespace dsymutil
} // end namespace llvm

#endif // LLVM_TOOLS_DSYMUTIL_PSEUDOPROBELINKER_H
