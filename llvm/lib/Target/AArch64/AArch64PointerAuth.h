//===-- AArch64PointerAuth.h -- Harden code using PAuth ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64POINTERAUTH_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64POINTERAUTH_H

#include "Utils/AArch64BaseInfo.h"

namespace llvm {
namespace AArch64PAuth {

/// PAuth key to be used with function pointers in .init_array and .fini_array.
constexpr AArch64PACKey::ID InitFiniKey = AArch64PACKey::IA;

/// Constant discriminator to be used with function pointers in .init_array and
/// .fini_array. The value is ptrauth_string_discriminator("init_fini")
constexpr unsigned InitFiniPointerConstantDiscriminator = 0xD9D4;

/// Controls whether, for PAUTH_EPILOGUE_ENTRY_SP, we use the hint-space
/// AUTI[AB]1716 instruction to authenticate the LR value, or whether we use
/// an AUTI[AB] LR, X16 instruction instead.
constexpr bool authenticatesLRViaX17(bool BranchProtectionPAuthLR,
                                     bool HasPAuth) {
  return BranchProtectionPAuthLR || !HasPAuth;
}

/// Variants of check performed on an authenticated pointer.
///
/// In cases such as authenticating the LR value when performing a tail call
/// or when re-signing a signed pointer with a different signing schema,
/// a failed authentication may not generate an exception on its own and may
/// create an authentication or signing oracle if not checked explicitly.
///
/// A number of check methods modify control flow in a similar way by
/// rewriting the code
///
/// ```
///   <authenticate LR>
///   <more instructions>
/// ```
///
/// as follows:
///
/// ```
///   <authenticate LR>
///   <method-specific checker>
/// on_fail:
///   brk <code>
/// on_success:
///   <more instructions>
///
/// ```
enum class AuthCheckMethod {
  /// Do not check the value at all
  None,

  /// Perform a load to a temporary register
  DummyLoad,

  /// Check by comparing bits 62 and 61 of the authenticated address.
  ///
  /// This method modifies control flow and inserts the following checker:
  ///
  /// ```
  ///   eor Xtmp, Xn, Xn, lsl #1
  ///   tbz Xtmp, #62, on_success
  /// ```
  HighBitsNoTBI,

  /// Check by comparing the authenticated value with an XPAC-ed one without
  /// using PAuth instructions not encoded as HINT. Can only be applied to LR.
  ///
  /// This method modifies control flow and inserts the following checker:
  ///
  /// ```
  ///   mov Xtmp, LR
  ///   xpaclri           ; encoded as "hint #7"
  ///   ; Note: at this point, the LR register contains the address as if
  ///   ; the authentication succeeded and the temporary register contains the
  ///   ; *real* result of authentication.
  ///   cmp Xtmp, LR
  ///   b.eq on_success
  /// ```
  XPACHint,

  /// Similar to XPACHint but using Armv8.3-only XPAC instruction, thus
  /// not restricted to LR:
  /// ```
  ///   mov Xtmp, Xn
  ///   xpac(i|d) Xn
  ///   cmp Xtmp, Xn
  ///   b.eq on_success
  /// ```
  XPAC,
};

/// Ways to check pointer authentication auth/resign failures.
enum class PtrauthCheckMode { Unchecked, Poison, Trap };

/// Control the emission of .cfi_set_ra_state, which replaces the
/// deprecated .cfi_negate_ra_state_with_pc [1].
///
/// The latter is fundamentally unable to express some program orders [2], as
/// the dwarf 'program' reads functions in a linear scan of their addresses to
/// reconstruct the state of the frame, whereas control flow may enter and exit
/// such regions arbitrarily (such as in hot-cold-split, and shrinkwrapped
/// fucntions), and thus the negate-based cfi is unable to encode the address of
/// the signing instruciton in all program orders.
///
/// Since .cfi_negate_ra_state is still sufficient for describing
/// ptrauth-returns=pauth, we default to using the new CFI only for PAuth_LR, as
/// DW_CFA_AARCH64_negate_ra_state has a smaller encoding than
/// DW_CFA_AARCH64_set_ra_state.
///
/// 1: https://github.com/ARM-software/abi-aa/pull/346
/// 2: https://github.com/ARM-software/abi-aa/issues/327
enum class SetRAStateMode {
  Never,   // Always use .cfi_negate_ra_state(_with_pc)
  PAuthLR, // Use .cfi_set_ra_state only for PAuth_LR
  Always,  // Use .cfi_set_ra_state for both PAuth and PAuth_LR
};

/// Returns the number of bytes added by checkAuthenticatedRegister.
unsigned getCheckerSizeInBytes(AuthCheckMethod Method);

} // end namespace AArch64PAuth
} // end namespace llvm

#endif
