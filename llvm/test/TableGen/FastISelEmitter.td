// RUN: llvm-tblgen --gen-fast-isel -I %p/../../include %s 2>&1 | FileCheck %s

include "llvm/Target/Target.td"

//===- Define the necessary boilerplate for our test target. --------------===//

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def R0 : Register<"r0"> { let Namespace = "MyTarget"; }
def R1 : Register<"r0"> { let Namespace = "MyTarget"; }
def GPR32M : RegisterClass<"MyTarget", [i32], 32, (add R0)>;
def GPR32MOp : RegisterOperand<GPR32M>;

def GPR32 : RegisterClass<"MyTarget", [i32], 32, (add R0, R1)>;
def GPR32Op : RegisterOperand<GPR32>;

class I<dag OOps, dag IOps, list<dag> Pat> : Instruction {
  let Namespace = "MyTarget";
  let OutOperandList = OOps;
  let InOperandList = IOps;
  let Pattern = Pat;
}

def HasA : Predicate<"Subtarget->hasA()">;

let Predicates = [HasA] in {

  def ADD : I<(outs GPR32Op:$rd), (ins GPR32Op:$rs, GPR32Op:$rt),
              [(set GPR32Op:$rd, (add GPR32Op:$rs, GPR32Op:$rt))]>;

  let FastISelShouldIgnore = 1 in
    def ADD_M : I<(outs GPR32MOp:$rd), (ins GPR32MOp:$rs, GPR32MOp:$rt),
                  [(set GPR32MOp:$rd, (add GPR32MOp:$rs, GPR32MOp:$rt))]>;
}

// CHECK-NOT: error: Duplicate predicate in FastISel table!
