//===-- VPLoopInfo.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines VPLoopInfo analysis and VPLoop class. VPLoopInfo is a
/// specialization of LoopInfoBase for VPBlockBase. VPLoops is a specialization
/// of LoopBase that is used to hold loop metadata from VPLoopInfo. Further
/// information can be found in VectorizationPlanner.rst.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VPLOOPINFO_H
#define LLVM_TRANSFORMS_VECTORIZE_VPLOOPINFO_H

#include "llvm/Analysis/LoopInfoImpl.h"

namespace llvm {
class VPBlockBase;

/// Hold analysis information for every loop detected by VPLoopInfo. It is an
/// instantiation of LoopBase.
class VPLoop : public LoopBase<VPBlockBase, VPLoop> {
private:
  friend class LoopInfoBase<VPBlockBase, VPLoop>;
  explicit VPLoop(VPBlockBase *VPB) : LoopBase<VPBlockBase, VPLoop>(VPB) {}
};

/// VPLoopInfo provides analysis of natural loop for VPBlockBase-based
/// Hierarchical CFG. It is a specialization of LoopInfoBase class.
// TODO: VPLoopInfo is initially computed on top of the VPlan plain CFG, which
// is the same as the incoming IR CFG. If it's more efficient than running the
// whole loop detection algorithm, we may want to create a mechanism to
// translate LoopInfo into VPLoopInfo. However, that would require significant
// changes in LoopInfoBase class.
typedef LoopInfoBase<VPBlockBase, VPLoop> VPLoopInfo;

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPLOOPINFO_H
