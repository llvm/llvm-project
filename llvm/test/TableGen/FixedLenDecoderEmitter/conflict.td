// RUN: llvm-tblgen -gen-disassembler -I %p/../../../include %s -o - 2>%t
// RUN: FileCheck %s < %t

include "llvm/Target/Target.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def R0 : Register<"r0"> { let Namespace = "MyTarget"; }
def GPR32 : RegisterClass<"MyTarget", [i32], 32, (add R0)>;

class I<dag OOps, dag IOps, list<dag> Pat>
  : Instruction {
  let Namespace = "MyTarget";
  let OutOperandList = OOps;
  let InOperandList = IOps;
  let Pattern = Pat;
  bits<32> Inst;
  bits<32> SoftFail;
}

def A : I<(outs GPR32:$dst), (ins GPR32:$src1), []> {
  let Size = 4;
  let Inst{31-0} = 0;
}
def B : I<(outs GPR32:$dst), (ins GPR32:$src1), []> {
  let Size = 4;
  let Inst{31-0} = 0;
}

// CHECK: Decoding Conflict:
// CHECK:   00000000000000000000000000000000
// CHECK:   ................................
// CHECK: A 00000000000000000000000000000000
// CHECK: B 00000000000000000000000000000000
