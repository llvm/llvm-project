//===- StandardInstrumentations.h ------------------------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This header defines a class that provides bookkeeping for all standard
/// (i.e in-tree) pass instrumentations.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_STANDARDINSTRUMENTATIONS_H
#define LLVM_PASSES_STANDARDINSTRUMENTATIONS_H

#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/PassTimingInfo.h"

namespace llvm {

/// This class provides an interface to register all the standard pass
/// instrumentations and manages their state (if any).
class StandardInstrumentations {
  TimePassesHandler TimePasses;

public:
  StandardInstrumentations() = default;

  void registerCallbacks(PassInstrumentationCallbacks &PIC);
};
} // namespace llvm

#endif
