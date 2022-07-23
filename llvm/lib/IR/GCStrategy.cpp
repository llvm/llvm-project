//===- GCStrategy.cpp - Garbage Collector Description ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the policy object GCStrategy which describes the
// behavior of a given garbage collector.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/GCStrategy.h"
#include "llvm/ADT/Twine.h"

using namespace llvm;

LLVM_INSTANTIATE_REGISTRY(GCRegistry)

GCStrategy::GCStrategy() = default;

std::unique_ptr<GCStrategy> llvm::getGCStrategy(const StringRef Name) {
  for (auto &S : GCRegistry::entries())
    if (S.getName() == Name)
      return S.instantiate();

  if (GCRegistry::begin() == GCRegistry::end()) {
    // In normal operation, the registry should not be empty.  There should
    // be the builtin GCs if nothing else.  The most likely scenario here is
    // that we got here without running the initializers used by the Registry
    // itself and it's registration mechanism.
    const std::string error =
        std::string("unsupported GC: ") + Name.str() +
        " (did you remember to link and initialize the library?)";
    report_fatal_error(Twine(error));
  } else
    report_fatal_error(Twine(std::string("unsupported GC: ") + Name.str()));
}
