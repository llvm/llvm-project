//=== lib/CodeGen/GlobalISel/AMDGPUCombinerHelper.h -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This contains common combine transformations that may be used in a combine
/// pass.
///
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"

using namespace llvm;

class AMDGPUCombinerHelper : public CombinerHelper {
public:
  using CombinerHelper::CombinerHelper;

  bool matchFoldableFneg(MachineInstr &MI, MachineInstr *&MatchInfo);
  void applyFoldableFneg(MachineInstr &MI, MachineInstr *&MatchInfo);
};
