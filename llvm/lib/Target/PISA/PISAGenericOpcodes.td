//===-- PISAGenericOpcodes.td - PISA Generic Opcodes -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

class PISAGenericInstruction : GenericInstruction {
  let Namespace = Global.NS;
}

def G_PISA_PARAM_SLOT : PISAGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins untyped_imm_0:$slot, untyped_imm_0:$offset,
                           variable_ops);
  let hasSideEffects = false;
}

// Target-specific generic instruction
class PISAUnaryInstruction : PISAGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1);
  let hasSideEffects = false;
}
class PISABinaryInstruction : PISAGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = false;
  let isCommutable = true;
}
class PISABinaryInstructionRndSat : PISAGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2, untyped_imm_0:$src3, untyped_imm_0:$src4);
  let hasSideEffects = false;
  let isCommutable = false;
}
class PISABinaryInstructionNanp : PISAGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2, untyped_imm_0:$nanp);
  let hasSideEffects = false;
  let isCommutable = true;
}
class PISATernaryInstruction : PISAGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2, type0:$src3);
  let hasSideEffects = false;
  let isCommutable = false;
}
class PISAReductionInstruction : PISAGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins untyped_imm_0:$op, type0:$src1, type1:$src2, type0:$src3);
  let hasSideEffects = false;
  let isCommutable = false;
}
class PISAReductionInstructionNanp : PISAGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins untyped_imm_0:$op, type0:$src1, type1:$src2, type0:$src3, untyped_imm_0:$nanp);
  let hasSideEffects = false;
  let isCommutable = false;
}
class PISATernaryInstructionRndSat : PISAGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2, type0:$src3, untyped_imm_0:$src4, untyped_imm_0:$src5);
  let hasSideEffects = false;
  let isCommutable = false;
}
class PISACompareInstruction : PISABinaryInstruction {
  let isCommutable = false;
}


class PISAConvertInstWithRnd : PISAGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src1, type2:$src2);
  let hasSideEffects = false;
}

class PISACasInstruction : PISAGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins untyped_imm_0:$src1, type0:$src2, type0:$src3, untyped_imm_0:$src4);
  let hasSideEffects = false;
  let isCommutable = false;
  let mayLoad = true;
  let mayStore = true;
}
class PISAFAtomInstruction : PISAGenericInstruction {
  let OutOperandList = (outs type0:$oldval);
  let InOperandList = (ins ptype1:$addr, type0:$val);
  let hasSideEffects = false;
  let isCommutable = false;
  let mayLoad = true;
  let mayStore = true;
}

// sel
def G_PISA_SELECT        : PISATernaryInstruction;
def pisa_select          : SDNode<"PISAISD::SELECT", SDTSelect>;
def : GINodeEquiv<G_PISA_SELECT, pisa_select>;
