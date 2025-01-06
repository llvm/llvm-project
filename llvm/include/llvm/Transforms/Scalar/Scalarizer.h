//===- Scalarizer.h --- Scalarize vector operations -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass converts vector operations into scalar operations (or, optionally,
/// operations on smaller vector widths), in order to expose optimization
/// opportunities on the individual scalar operations.
/// It is mainly intended for targets that do not have vector units, but it
/// may also be useful for revectorizing code to different vector widths.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_SCALARIZER_H
#define LLVM_TRANSFORMS_SCALAR_SCALARIZER_H

#include "llvm/IR/PassManager.h"
#include <optional>

namespace llvm {

class Function;
class FunctionPass;

struct ScalarizerPassOptions {
  /// Instruct the scalarizer pass to attempt to keep values of a minimum number
  /// of bits.

  /// Split vectors larger than this size into fragments, where each fragment is
  /// either a vector no larger than this size or a scalar.
  ///
  /// Instructions with operands or results of different sizes that would be
  /// split into a different number of fragments are currently left as-is.
  unsigned ScalarizeMinBits = 0;

  /// Allow the scalarizer pass to scalarize insertelement/extractelement with
  /// variable index.
  bool ScalarizeVariableInsertExtract = true;

  /// Allow the scalarizer pass to scalarize loads and store
  ///
  /// This is disabled by default because having separate loads and stores makes
  /// it more likely that the -combiner-alias-analysis limits will be reached.
  bool ScalarizeLoadStore = false;
};

class ScalarizerPass : public PassInfoMixin<ScalarizerPass> {
  ScalarizerPassOptions Options;

public:
  ScalarizerPass() = default;
  ScalarizerPass(const ScalarizerPassOptions &Options) : Options(Options) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  void setScalarizeVariableInsertExtract(bool Value) {
    Options.ScalarizeVariableInsertExtract = Value;
  }
  void setScalarizeLoadStore(bool Value) { Options.ScalarizeLoadStore = Value; }
  void setScalarizeMinBits(unsigned Value) { Options.ScalarizeMinBits = Value; }
};

/// Create a legacy pass manager instance of the Scalarizer pass
FunctionPass *createScalarizerPass(
    const ScalarizerPassOptions &Options = ScalarizerPassOptions());
}

#endif /* LLVM_TRANSFORMS_SCALAR_SCALARIZER_H */
