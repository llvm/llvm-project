
// RUN: llvm-tblgen -gen-emitter -I %p/../../include %s | \
// RUN:     FileCheck %s --check-prefix=ENCODER

include "llvm/Target/Target.td"

def archInstrInfo : InstrInfo { }

def arch : Target {
  let InstructionSet = archInstrInfo;
}

class PseudoInst<bits<16> Opcode> : Instruction {
  field bits<16> Inst;
  let Inst = Opcode;
  dag OutOperandList = (outs);
  dag InOperandList =  (ins);
  let Pattern = [];
  let isPseudo = 1;
}

def DummyInst : PseudoInst<0>;

// ENCODER-LABEL:   static const uint64_t InstBits[] = {
// ENCODER:         UINT64_C(0),
