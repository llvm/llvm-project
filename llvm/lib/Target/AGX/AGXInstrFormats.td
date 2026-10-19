//===-- AGXInstrFormats.td - AGX instruction formats -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Like AMDGPU, the AGX backend separates
//
//  * pseudo instructions (AGXPseudo), used by code generation, with precise
//    register classes, and
//  * real instructions (AGXReal), used by the MC layer, with a binary
//    encoding for one encoding family.
//
// AGXInstrInfo::pseudoToMCOpcode maps a pseudo to the real instruction of the
// subtarget's encoding family. Both have the same flattened operand list,
// except that real ALU destinations carry a trailing cache hint immediate.
//
// Instructions are up to 12 bytes long. Some instructions have a short and a
// long form; the long form sets a length bit and appends the high bits of the
// register and immediate fields. The `Inst` field always describes the long
// form, and the code emitter drops the extension when it is all zero.
//
//===----------------------------------------------------------------------===//

class AGXInst<dag outs, dag ins, string asm = "", list<dag> pattern = []>
    : Instruction {
  let Namespace = "AGX";
  let OutOperandList = outs;
  let InOperandList = ins;
  let AsmString = asm;
  let Pattern = pattern;
  let UseNamedOperandTable = 1;

  // Encoded length in bytes of the short and long forms. They are equal for
  // fixed length instructions.
  bits<4> ShortSize = 0;
  bits<4> LongSize = 0;
  // Position of the length bit for variable length instructions.
  bits<7> LengthBit = 15;

  // The instruction writes its results asynchronously (memory loads,
  // atomics, ...). A wait on its scoreboard slot is required before any
  // reader or writer of the results.
  bit IsAsync = 0;

  // The instruction reads or updates the execution mask nesting counter
  // (r0l) and the execution mask.
  bit IsExecMaskOp = 0;

  // The instruction belongs to a family with a common encoding layout.
  bit IsALU = 0;
  bit IsMemory = 0;

  let TSFlags{3-0} = ShortSize;
  let TSFlags{7-4} = LongSize;
  let TSFlags{14-8} = LengthBit;
  let TSFlags{15} = IsAsync;
  let TSFlags{16} = IsExecMaskOp;
  let TSFlags{17} = IsALU;
  let TSFlags{18} = IsMemory;
}

// Real instructions of the G13/G14 ("AGX2") encoding family.
class AGXReal<string mnemonic, dag outs, dag ins, string operands,
              int short, int long = short>
    : AGXInst<outs, ins, mnemonic#!if(!empty(operands), "", " "#operands)> {
  field bits<96> Inst;
  field bits<96> SoftFail = 0;

  let ShortSize = short;
  let LongSize = long;
  let Size = long;
  let DecoderNamespace = "G13";
  let Predicates = [HasAGX2Encoding];
  let AsmVariantName = "";
  let isCodeGenOnly = 0;
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;

  // Instructions without a length bit leave LengthBit at zero.
  let LengthBit = !if(!eq(short, long), 0, 15);
}

// Placeholder for "no real instruction in this encoding family".
def AGX_NO_REAL : AGXInst<(outs), (ins)> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

// Pseudo instructions used by code generation. G13Real names the real
// instruction for the AGX2 encoding family.
class AGXPseudo<dag outs, dag ins, list<dag> pattern = [], string asm = "">
    : AGXInst<outs, ins, asm, pattern> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
  let hasSideEffects = 0;
  let Size = 12;

  Instruction Pseudo = !cast<Instruction>(NAME);
  Instruction G13Real = AGX_NO_REAL;
  // Real instruction for the Apple9 encoding family, if any.
  Instruction Apple9Real = AGX_NO_REAL;
}

// Pseudo instructions without any real instruction: expanded before
// emission.
class AGXCodeGenPseudo<dag outs, dag ins, list<dag> pattern = []>
    : AGXInst<outs, ins, "", pattern> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
  let hasSideEffects = 0;
  let Size = 0;
}
