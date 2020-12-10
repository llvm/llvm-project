//===-- CallContext.h - Call Context Handler ---------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_PROFGEN_CALLCONTEXT_H
#define LLVM_TOOLS_LLVM_PROFGEN_CALLCONTEXT_H

#include "llvm/ProfileData/SampleProf.h"
#include <sstream>
#include <string>
#include <vector>

namespace llvm {
namespace sampleprof {

// Function name, LineLocation
typedef std::pair<std::string, LineLocation> FrameLocation;

typedef SmallVector<FrameLocation, 4> FrameLocationStack;

inline std::string getCallSite(const FrameLocation &Callsite) {
  std::string CallsiteStr = Callsite.first;
  CallsiteStr += ":";
  CallsiteStr += Twine(Callsite.second.LineOffset).str();
  if (Callsite.second.Discriminator > 0) {
    CallsiteStr += ".";
    CallsiteStr += Twine(Callsite.second.Discriminator).str();
  }
  return CallsiteStr;
}

// TODO: This operation is expansive. If it ever gets called multiple times we
// may think of making a class wrapper with internal states for it.
inline std::string getLocWithContext(const FrameLocationStack &Context) {
  std::ostringstream OContextStr;
  for (const auto &Callsite : Context) {
    if (OContextStr.str().size())
      OContextStr << " @ ";
    OContextStr << getCallSite(Callsite);
  }
  return OContextStr.str();
}

// Reverse call context, i.e., in the order of callee frames to caller frames,
// is useful during instruction printing or pseudo probe printing.
inline std::string
getReversedLocWithContext(const FrameLocationStack &Context) {
  std::ostringstream OContextStr;
  for (const auto &Callsite : reverse(Context)) {
    if (OContextStr.str().size())
      OContextStr << " @ ";
    OContextStr << getCallSite(Callsite);
  }
  return OContextStr.str();
}

} // end namespace sampleprof
} // end namespace llvm

#endif
