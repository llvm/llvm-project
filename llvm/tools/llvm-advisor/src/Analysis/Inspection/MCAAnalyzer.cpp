//===--- MCAAnalyzer.cpp - LLVM Advisor ----------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "Analysis/Inspection/MCAAnalyzer.h"
#include "llvm/Support/FileSystem.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
MCAAnalyzer::run(const CapabilityContext &Context) {
  if (Context.ObjectPath.empty() || !sys::fs::exists(Context.ObjectPath))
    return makeUnavailableResult(getCapabilityID(), Context.Unit.ID,
                                 "missing object artifact");

  return makeUnavailableResult(
      getCapabilityID(), Context.Unit.ID,
      "in-process MCA is disabled for this LLVM API version; use llvm-mca on "
      "the assembly artifact");
}
