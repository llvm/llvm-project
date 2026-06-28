//===-- SuperHInstrArithmetic.td - SuperH Arithmetic Instructions -*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes all of the arithmetic instructions in the SuperH
/// ISA.
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Namespace = "SH" in {
  def ADD_ir      :  Op_ir<0b0011100000000000, "add">;
  def ADD_rr      :  Op_rr<0b0011000000001100, "add">;
  def ADDC_rr     :  Op_rr<0b0011000000001110, "addc">;
  def ADDV_rr     :  Op_rr<0b0011000000001111, "addv">;
  def DIV0S_rr    :  Op_rr<0b0010000000000111, "div0s">;
  def DIV0U_rr    :     Op<0b0000000000011001, "div0u">;
  def DIV1_rr     :  Op_rr<0b0011000000000100, "div1">;
  def DIVS_rr0    : Op_rr0<0b0100000010010100, "divs">;
  def DIVU_rr0    : Op_rr0<0b0100000010000100, "divu">;
}