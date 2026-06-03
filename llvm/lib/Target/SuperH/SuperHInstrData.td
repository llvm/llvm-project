//===-- SuperHInstrData.td - SuperH Data Transfer Instructions -*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes all of the data transfer instructions in the SuperH
/// ISA.
///
//===----------------------------------------------------------------------===//



//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Namespace = "SH" in {
  def MOV_ir 		:    Op_ir<0b1110000000000000, "mov">;
  def MOV_rr 		:    Op_rr<0b0110000000000011, "mov">;
  def MOVB_rri	: Store_rr<0b0010000000000000, "mov.b">;
  def MOVW_rri	: Store_rr<0b0010000000000001, "mov.w">;
  def MOVL_rri	: Store_rr<0b0010000000000010, "mov.l">;
  def MOVB_rir	:  Load_rr<0b0110000000000000, "mov.b">;
  def MOVW_rir	:  Load_rr<0b0110000000000001, "mov.w">;
  def MOVL_rir	:  Load_rr<0b0110000000000010, "mov.l">;
}