//===- IntrinsicsAVR.td - Defines AVR intrinsics ------------*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the AVR-specific intrinsics.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "avr" in { // All intrinsics start with "llvm.avr.".

  // Simple void instruction intrinsics.
  def int_avr_nop : Intrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>;
  def int_avr_sei : Intrinsic<[], [], [IntrHasSideEffects]>;
  def int_avr_cli : Intrinsic<[], [], [IntrHasSideEffects]>;
  def int_avr_sleep : Intrinsic<[], [], [IntrHasSideEffects]>;
  def int_avr_wdr : Intrinsic<[], [], [IntrHasSideEffects]>;

  // Swap nibbles: returns i8 with high and low nibbles exchanged.
  def int_avr_swap : Intrinsic<[llvm_i8_ty], [llvm_i8_ty], [IntrNoMem]>;
}
