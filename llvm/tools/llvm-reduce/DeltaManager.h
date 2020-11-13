//===- DeltaManager.h - Runs Delta Passes to reduce Input -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file calls each specialized Delta pass in order to reduce the input IR
// file.
//
//===----------------------------------------------------------------------===//

#include "TestRunner.h"
#include "deltas/Delta.h"
#include "deltas/ReduceAliases.h"
#include "deltas/ReduceArguments.h"
#include "deltas/ReduceAttributes.h"
#include "deltas/ReduceBasicBlocks.h"
#include "deltas/ReduceFunctionBodies.h"
#include "deltas/ReduceFunctions.h"
#include "deltas/ReduceGlobalVars.h"
#include "deltas/ReduceInstructions.h"
#include "deltas/ReduceMetadata.h"
#include "deltas/ReduceOperandBundles.h"
#include "deltas/ReduceSpecialGlobals.h"

namespace llvm {

// TODO: Add CLI option to run only specified Passes (for unit tests)
inline void runDeltaPasses(TestRunner &Tester) {
  reduceSpecialGlobalsDeltaPass(Tester);
  reduceAliasesDeltaPass(Tester);
  reduceFunctionBodiesDeltaPass(Tester);
  reduceFunctionsDeltaPass(Tester);
  reduceBasicBlocksDeltaPass(Tester);
  reduceGlobalsDeltaPass(Tester);
  reduceMetadataDeltaPass(Tester);
  reduceArgumentsDeltaPass(Tester);
  reduceInstructionsDeltaPass(Tester);
  reduceOperandBundesDeltaPass(Tester);
  reduceAttributesDeltaPass(Tester);
  // TODO: Implement the remaining Delta Passes
}

} // namespace llvm
