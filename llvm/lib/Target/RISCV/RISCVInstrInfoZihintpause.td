//===-- RISCVInstrInfoZihintpause.td -----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

let Predicates = [HasStdExtZihintpause] in {
  def : Pat<(int_riscv_pause), (FENCE 0x1, 0x0)>;
}
