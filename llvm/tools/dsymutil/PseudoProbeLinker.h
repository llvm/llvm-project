//===----------------------------------------------------------------------===//
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

/// The PseudoProbeLinker collects and concatenates pseudo probe sections
/// __probes and __probe_descs from the debug map objects and writes them as
/// binary blobs into the .dSYM bundle.
class PseudoProbeLinker {
  const LinkOptions &Options;
  std::string Probes;
  std::string ProbeDescs;

public:
  PseudoProbeLinker(const LinkOptions &Options) : Options(Options) {}

  Error collect(const object::ObjectFile &Obj);

  Error emit(const Triple &TheTriple) const;

  bool empty() const { return Probes.empty() && ProbeDescs.empty(); }
  StringRef getProbes() const { return Probes; }
  StringRef getProbeDescs() const { return ProbeDescs; }
};

} // end namespace dsymutil
} // end namespace llvm

#endif // LLVM_TOOLS_DSYMUTIL_PSEUDOPROBELINKER_H
