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

class AArch64TargetLowering;

class Function;
class CallBase;
class AttributeList;

/// SMEAttrs is a utility class to parse the SME ACLE attributes on functions.
/// It helps determine a function's requirements for PSTATE.ZA and PSTATE.SM.
class SMEAttrs {
  unsigned Bitmask = Normal;

public:
  enum class StateValue {
    None = 0,
    In = 1,        // aarch64_in_zt0
    Out = 2,       // aarch64_out_zt0
    InOut = 3,     // aarch64_inout_zt0
    Preserved = 4, // aarch64_preserves_zt0
    New = 5        // aarch64_new_zt0
  };

  // Enum with bitmasks for each individual SME feature.
  enum Mask {
    Normal = 0,
    SM_Enabled = 1 << 0,      // aarch64_pstate_sm_enabled
    SM_Compatible = 1 << 1,   // aarch64_pstate_sm_compatible
    SM_Body = 1 << 2,         // aarch64_pstate_sm_body
    SME_ABI_Routine = 1 << 3, // Used for SME ABI routines to avoid lazy saves
    ZA_State_Agnostic = 1 << 4,
    ZT0_Undef = 1 << 5, // Use to mark ZT0 as undef to avoid spills
    ZA_Shift = 6,
    ZA_Mask = 0b111 << ZA_Shift,
    ZT0_Shift = 9,
    ZT0_Mask = 0b111 << ZT0_Shift,
    CallSiteFlags_Mask = ZT0_Undef
  };

  SMEAttrs() = default;
  SMEAttrs(unsigned Mask) { set(Mask); }
  SMEAttrs(const Function &F, const AArch64TargetLowering *TLI = nullptr)
      : SMEAttrs(F.getAttributes()) {
    if (TLI)
      addKnownFunctionAttrs(F.getName(), *TLI);
  }
  SMEAttrs(const AttributeList &L);
  SMEAttrs(StringRef FuncName, const AArch64TargetLowering &TLI) {
    addKnownFunctionAttrs(FuncName, TLI);
  };

  void set(unsigned M, bool Enable = true) {
    if (Enable)
      Bitmask |= M;
    else
      Bitmask &= ~M;
#ifndef NDEBUG
    validate();
#endif
  }

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

  // Interfaces to query ZA
  static StateValue decodeZAState(unsigned Bitmask) {
    return static_cast<StateValue>((Bitmask & ZA_Mask) >> ZA_Shift);
  }
  static unsigned encodeZAState(StateValue S) {
    return static_cast<unsigned>(S) << ZA_Shift;
  }

  bool isNewZA() const { return decodeZAState(Bitmask) == StateValue::New; }
  bool isInZA() const { return decodeZAState(Bitmask) == StateValue::In; }
  bool isOutZA() const { return decodeZAState(Bitmask) == StateValue::Out; }
  bool isInOutZA() const { return decodeZAState(Bitmask) == StateValue::InOut; }
  bool isPreservesZA() const {
    return decodeZAState(Bitmask) == StateValue::Preserved;
  }
  bool sharesZA() const {
    StateValue State = decodeZAState(Bitmask);
    return State == StateValue::In || State == StateValue::Out ||
           State == StateValue::InOut || State == StateValue::Preserved;
  }
  bool hasAgnosticZAInterface() const { return Bitmask & ZA_State_Agnostic; }
  bool hasSharedZAInterface() const { return sharesZA() || sharesZT0(); }
  bool hasPrivateZAInterface() const {
    return !hasSharedZAInterface() && !hasAgnosticZAInterface();
  }
  bool hasZAState() const { return isNewZA() || sharesZA(); }
  bool isSMEABIRoutine() const { return Bitmask & SME_ABI_Routine; }

  // Interfaces to query ZT0 State
  static StateValue decodeZT0State(unsigned Bitmask) {
    return static_cast<StateValue>((Bitmask & ZT0_Mask) >> ZT0_Shift);
  }
  static unsigned encodeZT0State(StateValue S) {
    return static_cast<unsigned>(S) << ZT0_Shift;
  }

  bool isNewZT0() const { return decodeZT0State(Bitmask) == StateValue::New; }
  bool isInZT0() const { return decodeZT0State(Bitmask) == StateValue::In; }
  bool isOutZT0() const { return decodeZT0State(Bitmask) == StateValue::Out; }
  bool isInOutZT0() const {
    return decodeZT0State(Bitmask) == StateValue::InOut;
  }
  bool isPreservesZT0() const {
    return decodeZT0State(Bitmask) == StateValue::Preserved;
  }
  bool hasUndefZT0() const { return Bitmask & ZT0_Undef; }
  bool sharesZT0() const {
    StateValue State = decodeZT0State(Bitmask);
    return State == StateValue::In || State == StateValue::Out ||
           State == StateValue::InOut || State == StateValue::Preserved;
  }
  bool hasZT0State() const { return isNewZT0() || sharesZT0(); }

  SMEAttrs operator|(SMEAttrs Other) const {
    SMEAttrs Merged(*this);
    Merged.set(Other.Bitmask);
    return Merged;
  }

  SMEAttrs withoutPerCallsiteFlags() const {
    return (Bitmask & ~CallSiteFlags_Mask);
  }

  bool operator==(SMEAttrs const &Other) const {
    return Bitmask == Other.Bitmask;
  }

private:
  void addKnownFunctionAttrs(StringRef FuncName,
                             const AArch64TargetLowering &TLI);
  void validate() const;
};

/// SMECallAttrs is a utility class to hold the SMEAttrs for a callsite. It has
/// interfaces to query whether a streaming mode change or lazy-save mechanism
/// is required when going from one function to another (e.g. through a call).
class SMECallAttrs {
  SMEAttrs CallerFn;
  SMEAttrs CalledFn;
  SMEAttrs Callsite;
  bool IsIndirect = false;

public:
  SMECallAttrs(SMEAttrs Caller, SMEAttrs Callee,
               SMEAttrs Callsite = SMEAttrs::Normal)
      : CallerFn(Caller), CalledFn(Callee), Callsite(Callsite) {}

  SMECallAttrs(const CallBase &CB, const AArch64TargetLowering *TLI);

  SMEAttrs &caller() { return CallerFn; }
  SMEAttrs &callee() { return IsIndirect ? Callsite : CalledFn; }
  SMEAttrs &callsite() { return Callsite; }
  SMEAttrs const &caller() const { return CallerFn; }
  SMEAttrs const &callee() const {
    return const_cast<SMECallAttrs *>(this)->callee();
  }
  SMEAttrs const &callsite() const { return Callsite; }

  /// \return true if a call from Caller -> Callee requires a change in
  /// streaming mode.
  bool requiresSMChange() const;

  bool requiresLazySave() const {
    return caller().hasZAState() && callee().hasPrivateZAInterface() &&
           !callee().isSMEABIRoutine();
  }

  bool requiresPreservingZT0() const {
    return caller().hasZT0State() && !callsite().hasUndefZT0() &&
           !callee().sharesZT0() && !callee().hasAgnosticZAInterface();
  }

  bool requiresDisablingZABeforeCall() const {
    return caller().hasZT0State() && !caller().hasZAState() &&
           callee().hasPrivateZAInterface() && !callee().isSMEABIRoutine();
  }

  bool requiresEnablingZAAfterCall() const {
    return requiresLazySave() || requiresDisablingZABeforeCall();
  }

  bool requiresPreservingAllZAState() const {
    return caller().hasAgnosticZAInterface() &&
           !callee().hasAgnosticZAInterface() && !callee().isSMEABIRoutine();
  }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AARCH64_UTILS_AARCH64SMEATTRIBUTES_H
