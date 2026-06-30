//===- tools/dsymutil/PseudoProbe.h ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_DSYMUTIL_PSEUDOPROBE_H
#define LLVM_TOOLS_DSYMUTIL_PSEUDOPROBE_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
class MCStreamer;
namespace object {
class MachOObjectFile;
}

namespace dsymutil {

inline constexpr StringRef PseudoProbeSegmentName = "__LLVM";
inline constexpr StringRef PseudoProbeSectionName = "__probes";
inline constexpr StringRef PseudoProbeDescSectionName = "__probe_descs";

class PseudoProbeCollector {
public:
  explicit PseudoProbeCollector(MCStreamer &Streamer) : Streamer(Streamer) {}

  /// Append \p Obj's \c __probes / \c __probe_descs sections to the merged
  /// output sections. Safe to call with objects that have no probe sections.
  void collectFromObject(const object::MachOObjectFile &Obj);

private:
  void emit(StringRef SecName, StringRef Contents, uint32_t Alignment);

  MCStreamer &Streamer;
};

} // end namespace dsymutil
} // end namespace llvm

#endif // LLVM_TOOLS_DSYMUTIL_PSEUDOPROBE_H
