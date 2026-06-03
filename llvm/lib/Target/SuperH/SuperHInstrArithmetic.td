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
  def ADD_ir : Op_rr<0b0011100000000000, "add">;
  def ADD_rr : Op_rr<0b0011000000001100, "add">;
}