// RUN: not llvm-tblgen -gen-disassembler -I %p/../../../include %s -o - 2>%t
// RUN: FileCheck %s < %t

include "llvm/Target/Target.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def R0 : Register<"r0"> { let Namespace = "MyTarget"; }
def GPR32 : RegisterClass<"MyTarget", [i32], 32, (add R0)>;

def X0 : Register<"x0"> { let Namespace = "MyTarget"; }
def GPR64 : RegisterClass<"MyTarget", [i64], 64, (add X0)>;

class I<dag OOps, dag IOps, list<dag> Pat>
  : Instruction {
  let Namespace = "MyTarget";
  let OutOperandList = OOps;
  let InOperandList = IOps;
  let Pattern = Pat;
  let Size = 4;
  bits<32> Inst;
  bits<32> SoftFail;
}

// Assume there is a 2 bit encoding for the dst and src register.
def A : I<(outs GPR32:$dst), (ins GPR32:$src1), []> {
  bits<2> dst;
  bits<2> src1;
  let Inst{31...4} = 0;
  let Inst{1...0} = dst;
  let Inst{3...2} = src1;
}

def B : I<(outs GPR32:$dst), (ins GPR32:$src1), []> {
  bits<2> dst;
  bits<2> src1;
  let Inst{31...4} = 0;
  let Inst{1...0} = dst;
  let Inst{3...2} = src1;
}

def C : I<(outs GPR64:$dst), (ins GPR64:$src1), []> {
  bits<2> dst;
  bits<2> src1;
  let Inst{31...4} = 1;
  let Inst{1...0} = dst;
  let Inst{3...2} = src1;
}

def D : I<(outs GPR64:$dst), (ins GPR64:$src1), []> {
  bits<2> dst;
  bits<2> src1;
  let Inst{31...4} = 1;
  let Inst{1...0} = dst;
  let Inst{3...2} = src1;
}

// CHECK: Decoding Conflict:
// CHECK:     ................................
// CHECK:     0000000000000000000000000000....
// CHECK:     0000000000000000000000000000____  A
// CHECK:     0000000000000000000000000000____  B

// CHECK: Decoding Conflict:
// CHECK:     ................................
// CHECK:     0000000000000000000000000001....
// CHECK:     0000000000000000000000000001____  C
// CHECK:     0000000000000000000000000001____  D

// CHECK: Decoding conflict encountered
