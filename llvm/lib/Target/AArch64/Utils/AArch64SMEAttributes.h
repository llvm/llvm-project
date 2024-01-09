//===-- AArch64SMEAttributes.h - Helper for interpreting SME attributes -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_UTILS_AARCH64SMEATTRIBUTES_H
#define LLVM_LIB_TARGET_AARCH64_UTILS_AARCH64SMEATTRIBUTES_H

#include "llvm/IR/Function.h"

namespace llvm {

class Function;
class CallBase;
class AttributeList;

/// SMEAttrs is a utility class to parse the SME ACLE attributes on functions.
/// It helps determine a function's requirements for PSTATE.ZA and PSTATE.SM. It
/// has interfaces to query whether a streaming mode change or lazy-save
/// mechanism is required when going from one function to another (e.g. through
/// a call).
class SMEAttrs {
  unsigned Bitmask;

public:
  // Enum with bitmasks for each individual SME feature.
  enum Mask {
    Normal = 0,
    SM_Enabled = 1 << 0,     // aarch64_pstate_sm_enabled
    SM_Compatible = 1 << 1,  // aarch64_pstate_sm_compatible
    SM_Body = 1 << 2,        // aarch64_pstate_sm_body
    ZA_Shared = 1 << 3,      // aarch64_pstate_sm_shared
    ZA_New = 1 << 4,         // aarch64_pstate_sm_new
    ZA_Preserved = 1 << 5,   // aarch64_pstate_sm_preserved
    ZA_NoLazySave = 1 << 6,  // Used for SME ABI routines to avoid lazy saves
    ZT0_New = 1 << 7,        // aarch64_sme_zt0_new
    ZT0_In = 1 << 8,         // aarch64_sme_zt0_in
    ZT0_Out = 1 << 9,        // aarch64_sme_zt0_out
    ZT0_InOut = 1 << 10,     // aarch64_sme_zt0_inout
    ZT0_Preserved = 1 << 11, // aarch64_sme_zt0_preserved
    All = ZT0_Preserved - 1
  };

  SMEAttrs(unsigned Mask = Normal) : Bitmask(0) { set(Mask); }
  SMEAttrs(const Function &F) : SMEAttrs(F.getAttributes()) {}
  SMEAttrs(const CallBase &CB);
  SMEAttrs(const AttributeList &L);
  SMEAttrs(StringRef FuncName);

  void set(unsigned M, bool Enable = true);

  // Interfaces to query PSTATE.SM
  bool hasStreamingBody() const { return Bitmask & SM_Body; }
  bool hasStreamingInterface() const { return Bitmask & SM_Enabled; }
  bool hasStreamingInterfaceOrBody() const {
    return hasStreamingBody() || hasStreamingInterface();
  }
  bool hasStreamingCompatibleInterface() const {
    return Bitmask & SM_Compatible;
  }
  bool hasNonStreamingInterface() const {
    return !hasStreamingInterface() && !hasStreamingCompatibleInterface();
  }
  bool hasNonStreamingInterfaceAndBody() const {
    return hasNonStreamingInterface() && !hasStreamingBody();
  }

  /// \return true if a call from Caller -> Callee requires a change in
  /// streaming mode.
  /// If \p BodyOverridesInterface is true and Callee has a streaming body,
  /// then requiresSMChange considers a call to Callee as having a Streaming
  /// interface. This can be useful when considering e.g. inlining, where we
  /// explicitly want the body to overrule the interface (because after inlining
  /// the interface is no longer relevant).
  std::optional<bool>
  requiresSMChange(const SMEAttrs &Callee,
                   bool BodyOverridesInterface = false) const;

  // Interfaces to query PSTATE.ZA
  bool hasNewZABody() const { return Bitmask & ZA_New; }
  bool hasSharedZAInterface() const { return Bitmask & ZA_Shared; }
  bool hasPrivateZAInterface() const { return !hasSharedZAInterface(); }
  bool preservesZA() const { return Bitmask & ZA_Preserved; }
  bool hasZAState() const {
    return hasNewZABody() || hasSharedZAInterface();
  }
  bool requiresLazySave(const SMEAttrs &Callee) const {
    return hasZAState() && Callee.hasPrivateZAInterface() &&
           !(Callee.Bitmask & ZA_NoLazySave);
  }

  // Interfaces to query ZT0 State
  bool hasNewZT0Body() const { return Bitmask & ZT0_New; }
  bool isZT0In() const { return Bitmask & ZT0_In; }
  bool isZT0Out() const { return Bitmask & ZT0_Out; }
  bool isZT0InOut() const { return Bitmask & ZT0_InOut; }
  bool preservesZT0() const { return Bitmask & ZT0_Preserved; }
  bool sharesZT0() const {
    return Bitmask & (ZT0_In | ZT0_Out | ZT0_InOut | ZT0_Preserved);
  }
  bool hasZT0State() const { return hasNewZT0Body() || sharesZT0(); }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AARCH64_UTILS_AARCH64SMEATTRIBUTES_H
