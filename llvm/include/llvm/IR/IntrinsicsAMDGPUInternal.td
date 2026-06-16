//===- IntrinsicsAMDGPUInternal.td - Internal intrinsics -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines AMDGPU intrinsics that are internal to the backend. They
// are inserted by backend passes (not intended to be written by frontends or
// used as a stable, documented interface) and may be changed or removed without
// notice. Keeping them separate from IntrinsicsAMDGPU.td makes that contract
// explicit and avoids advertising them as a general-purpose facility.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "amdgcn" in {

// Void marker that pins its argument (intended to be inserted by
// AMDGPUPromoteAlloca): it is lowered to a SI_VGPR_PIN pseudo whose vreg is
// later marked not-spillable by SIPinVGPR, keeping that value VGPR-resident
// under pressure. It returns void and simply applies to its operand, so there
// is no result for surrounding code to flow through (no pass-through behavior).
//
// The pin is binding: the allocator will not spill a pinned value (it spills
// other values instead), and if the pinned values cannot fit the VGPR budget
// allocation fails rather than spilling them. The producer must keep the
// simultaneously-pinned set within budget.
//
// IntrInaccessibleMemOnly (rather than IntrNoMem) gives it just enough of a
// side effect that it is not deleted as dead before it reaches the backend,
// while still not aliasing any real memory.
//
// This is an implementation detail of how the backend keeps promoted allocas in
// VGPRs; it is deliberately not a user-facing knob for tweaking register
// allocation. The user-facing way to request VGPR residency is tied to allocas
// (handled separately); this intrinsic is only the lowering vehicle for it.
//
// The argument must not be (vector-)i1: a lane mask cannot be made
// VGPR-resident. This is enforced by the IR verifier.
def int_amdgcn_internal_vgpr_pin :
  DefaultAttrsIntrinsic<[], [llvm_any_ty],
                        [IntrInaccessibleMemOnly]>;

} // TargetPrefix = "amdgcn"
