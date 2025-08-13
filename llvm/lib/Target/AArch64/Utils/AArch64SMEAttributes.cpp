//===-- AArch64SMEAttributes.cpp - Helper for interpreting SME attributes -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64SMEAttributes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/RuntimeLibcalls.h"
#include <cassert>

using namespace llvm;

void SMEAttrs::validate() const {
  // Streaming Mode Attrs
  assert(!(hasStreamingInterface() && hasStreamingCompatibleInterface()) &&
         "SM_Enabled and SM_Compatible are mutually exclusive");

  // ZA Attrs
  assert(!(isNewZA() && (Bitmask & SME_ABI_Routine)) &&
         "ZA_New and SME_ABI_Routine are mutually exclusive");

  assert(
      (isNewZA() + isInZA() + isOutZA() + isInOutZA() + isPreservesZA()) <= 1 &&
      "Attributes 'aarch64_new_za', 'aarch64_in_za', 'aarch64_out_za', "
      "'aarch64_inout_za' and 'aarch64_preserves_za' are mutually exclusive");

  // ZT0 Attrs
  assert(
      (isNewZT0() + isInZT0() + isOutZT0() + isInOutZT0() + isPreservesZT0()) <=
          1 &&
      "Attributes 'aarch64_new_zt0', 'aarch64_in_zt0', 'aarch64_out_zt0', "
      "'aarch64_inout_zt0' and 'aarch64_preserves_zt0' are mutually exclusive");

  assert(!(hasAgnosticZAInterface() && hasSharedZAInterface()) &&
         "Function cannot have a shared-ZA interface and an agnostic-ZA "
         "interface");
}

SMEAttrs::SMEAttrs(const AttributeList &Attrs) {
  Bitmask = 0;
  if (Attrs.hasFnAttr("aarch64_pstate_sm_enabled"))
    Bitmask |= SM_Enabled;
  if (Attrs.hasFnAttr("aarch64_pstate_sm_compatible"))
    Bitmask |= SM_Compatible;
  if (Attrs.hasFnAttr("aarch64_pstate_sm_body"))
    Bitmask |= SM_Body;
  if (Attrs.hasFnAttr("aarch64_za_state_agnostic"))
    Bitmask |= ZA_State_Agnostic;
  if (Attrs.hasFnAttr("aarch64_zt0_undef"))
    Bitmask |= ZT0_Undef;
  if (Attrs.hasFnAttr("aarch64_in_za"))
    Bitmask |= encodeZAState(StateValue::In);
  if (Attrs.hasFnAttr("aarch64_out_za"))
    Bitmask |= encodeZAState(StateValue::Out);
  if (Attrs.hasFnAttr("aarch64_inout_za"))
    Bitmask |= encodeZAState(StateValue::InOut);
  if (Attrs.hasFnAttr("aarch64_preserves_za"))
    Bitmask |= encodeZAState(StateValue::Preserved);
  if (Attrs.hasFnAttr("aarch64_new_za"))
    Bitmask |= encodeZAState(StateValue::New);
  if (Attrs.hasFnAttr("aarch64_in_zt0"))
    Bitmask |= encodeZT0State(StateValue::In);
  if (Attrs.hasFnAttr("aarch64_out_zt0"))
    Bitmask |= encodeZT0State(StateValue::Out);
  if (Attrs.hasFnAttr("aarch64_inout_zt0"))
    Bitmask |= encodeZT0State(StateValue::InOut);
  if (Attrs.hasFnAttr("aarch64_preserves_zt0"))
    Bitmask |= encodeZT0State(StateValue::Preserved);
  if (Attrs.hasFnAttr("aarch64_new_zt0"))
    Bitmask |= encodeZT0State(StateValue::New);
}

void SMEAttrs::addKnownFunctionAttrs(StringRef FuncName,
                                     const TargetLowering &TLI) {
  // If the function name does not start with a _ or #_ is not a builtin.
  if (!FuncName.starts_with('_') && !FuncName.starts_with("#_"))
    return;

  struct SMERoutineAttr {
    RTLIB::Libcall LC{RTLIB::UNKNOWN_LIBCALL};
    unsigned Attrs{SMEAttrs::Normal};
  };

  static constexpr unsigned SMCompatiableABIRoutine =
      SMEAttrs::SM_Compatible | SMEAttrs::SME_ABI_Routine;
  static constexpr unsigned SMCompatiableABIRoutineInZA =
      SMCompatiableABIRoutine | encodeZAState(StateValue::In);

  // Table of SME routine -> Known attributes.
  static constexpr SMERoutineAttr SMERoutineAttrs[]{
      {RTLIB::SMEABI_SME_STATE, SMCompatiableABIRoutine},
      {RTLIB::SMEABI_TPIDR2_SAVE, SMCompatiableABIRoutine},
      {RTLIB::SMEABI_GET_CURRENT_VG, SMCompatiableABIRoutine},
      {RTLIB::SMEABI_SME_STATE_SIZE, SMCompatiableABIRoutine},
      {RTLIB::SMEABI_SME_SAVE, SMCompatiableABIRoutine},
      {RTLIB::SMEABI_SME_RESTORE, SMCompatiableABIRoutine},
      {RTLIB::SMEABI_ZA_DISABLE, SMCompatiableABIRoutineInZA},
      {RTLIB::SMEABI_TPIDR2_RESTORE, SMCompatiableABIRoutineInZA},
      {RTLIB::SC_MEMCPY, SMEAttrs::SM_Compatible},
      {RTLIB::SC_MEMMOVE, SMEAttrs::SM_Compatible},
      {RTLIB::SC_MEMSET, SMEAttrs::SM_Compatible},
      {RTLIB::SC_MEMCHR, SMEAttrs::SM_Compatible},
  };

  unsigned KnownAttrs = SMEAttrs::Normal;
  for (auto [LC, Attrs] : SMERoutineAttrs) {
    if (TLI.getLibcallName(LC) == FuncName) {
      KnownAttrs = Attrs;
      break;
    }
  }

  set(KnownAttrs);
}

bool SMECallAttrs::requiresSMChange() const {
  if (callee().hasStreamingCompatibleInterface())
    return false;

  // Both non-streaming
  if (caller().hasNonStreamingInterfaceAndBody() &&
      callee().hasNonStreamingInterface())
    return false;

  // Both streaming
  if (caller().hasStreamingInterfaceOrBody() &&
      callee().hasStreamingInterface())
    return false;

  return true;
}

SMECallAttrs::SMECallAttrs(const CallBase &CB, const TargetLowering *TLI)
    : CallerFn(*CB.getFunction()), CalledFn(SMEAttrs::Normal),
      Callsite(CB.getAttributes()), IsIndirect(CB.isIndirectCall()) {
  if (auto *CalledFunction = CB.getCalledFunction())
    CalledFn = SMEAttrs(*CalledFunction, TLI);

  // FIXME: We probably should not allow SME attributes on direct calls but
  // clang duplicates streaming mode attributes at each callsite.
  assert((IsIndirect ||
          ((Callsite.withoutPerCallsiteFlags() | CalledFn) == CalledFn)) &&
         "SME attributes at callsite do not match declaration");
}
