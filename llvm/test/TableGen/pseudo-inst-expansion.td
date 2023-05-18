// RUN: llvm-tblgen -gen-pseudo-lowering -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def TestTargetInstrInfo : InstrInfo;

def TestTarget : Target {
  let InstructionSet = TestTargetInstrInfo;
}

def REG : Register<"REG">;
def GPR : RegisterClass<"TestTarget", [i32], 32, (add REG)>;

class SysReg<bits<12> op> {
  bits<12> Encoding = op;
}
def SR : SysReg<0b111100001111>;

class Pseudo<dag outs, dag ins, list<dag> pattern>
    : Instruction {
  dag OutOperandList = outs;
  dag InOperandList = ins;
  let Pattern = pattern;
  let isPseudo = 1;
}

def INSTR : Instruction {
  let OutOperandList = (outs GPR:$rd);
  let InOperandList = (ins i32imm:$val);
  let Pattern = [];
}

def PSEUDO : Pseudo<(outs GPR:$rd), (ins),
                    [(set GPR:$rd, (i32 SR.Encoding))]>,
             PseudoInstExpansion<(INSTR GPR:$rd, SR.Encoding)>;

// CHECK: .addOperand(MCOperand::createImm(3855));
