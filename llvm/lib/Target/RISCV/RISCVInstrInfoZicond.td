//===-- RISCVInstrInfoZicond.td ----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard Integer
// Conditional operations extension (Zicond).
// This version is still experimental as the 'Zicond' extension hasn't been
// ratified yet. It is based on v1.0-rc1 of the specification.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZicond] in {
def CZERO_EQZ : ALU_rr<0b0000111, 0b101, "czero.eqz">,
                Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def CZERO_NEZ : ALU_rr<0b0000111, 0b111, "czero.nez">,
                Sched<[WriteIALU, ReadIALU, ReadIALU]>;
} // Predicates = [HasStdExtZicond]
