//===- XtensaOperands.td - Xtensa instruction operands -------*- tblgen-*--===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Immediate operands with a shared generic render method.
class ImmAsmOperand<string name> : AsmOperandClass {
  let Name = name;
  let RenderMethod = "addImmOperands";
  let DiagnosticType = !strconcat("Invalid", name);
}

class Immediate<ValueType vt, code pred, string asmop>
  : Operand<vt>, ImmLeaf<vt, pred> {
  let PrintMethod = "print"#asmop;
  let ParserMatchClass = !cast<AsmOperandClass>(asmop);
}

// imm8 predicate - Immediate in the range [-128,127]
def Imm8_AsmOperand : ImmAsmOperand<"Imm8">;
def imm8 : Immediate<i32, [{ return Imm >= -128 && Imm <= 127; }], "Imm8_AsmOperand"> {
  let EncoderMethod = "getImm8OpValue";
}

// imm8_sh8 predicate - Immediate in the range [-32768,32512] with (bits[7-0] == 0)
// imm8 value left shifted by 8 bits
def Imm8_sh8_AsmOperand : ImmAsmOperand<"Imm8_sh8">;
def imm8_sh8 : Immediate<i32, [{ return Imm >= -32768 && Imm <= 32512 && ((Imm & 0xFF) == 0); }],
                        "Imm8_sh8_AsmOperand"> {
  let EncoderMethod = "getImm8_sh8OpValue";
}
