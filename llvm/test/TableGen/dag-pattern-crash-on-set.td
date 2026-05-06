// RUN: not llvm-tblgen -gen-dag-isel -I %p/../../include -o /dev/null %s 2>&1 | FileCheck %s

include "llvm/Target/Target.td"

def MyTargetInstrInfo : InstrInfo;
def MyTarget : Target {
  let InstructionSet = MyTargetInstrInfo;
}

def R0 : Register<"r0">;
def GPR : RegisterClass<"MyTarget", [i32], 32, (add R0)>;

def LOAD : Instruction {
  let Size = 2;
  let OutOperandList = (outs GPR:$dst);
  let InOperandList = (ins GPR:$addr);
  let AsmString = "movimm $dst, $addr";
}

// CHECK: [[@LINE+1]]:5: error: In CrashPat: unknown node type 'set' in input pattern
def CrashPat : Pat <
  (set R0, (load GPR:$addr)),
  (LOAD $addr)
>;

