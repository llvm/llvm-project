//===- SkeletonEmitter.cpp - Skeleton TableGen backend          -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This Tablegen backend emits ...
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <algorithm>
#include <set>
#include <string>
#include <vector>

#define DEBUG_TYPE "skeleton-emitter"

using namespace llvm;

namespace {

// Any helper data structures can be defined here. Some backends use
// structs to collect information from the records.

class SkeletonEmitter {
private:
  RecordKeeper &Records;

public:
  SkeletonEmitter(RecordKeeper &RK) : Records(RK) {}

  void run(raw_ostream &OS);
}; // emitter class

} // anonymous namespace

void SkeletonEmitter::run(raw_ostream &OS) {
  emitSourceFileHeader("Skeleton data structures", OS);

  (void)Records; // To suppress unused variable warning; remove on use.
}

namespace llvm {

// The only thing that should be in the llvm namespace is the
// emitter entry point function.

void EmitSkeleton(RecordKeeper &RK, raw_ostream &OS) {
  // Instantiate the emitter class and invoke run().
  SkeletonEmitter(RK).run(OS);
}

} // namespace llvm
