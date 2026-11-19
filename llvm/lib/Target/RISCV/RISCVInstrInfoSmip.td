//===-- RISCVInstrInfoSmip.td ------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtSmip] in
def MIPOPRET : Priv<"mipopret", 0b0011000>, Sched<[]> {
  let rd = 0;
  let rs1 = 0;
  let rs2 = 0b01000;
}

let Predicates = [HasStdExtSsip] in
def SIPOPRET : Priv<"sipopret", 0b0001000>, Sched<[]> {
  let rd = 0;
  let rs1 = 0;
  let rs2 = 0b01000;
}
