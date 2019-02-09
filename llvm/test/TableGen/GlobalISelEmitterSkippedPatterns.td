// RUN: llvm-tblgen -warn-on-skipped-patterns -gen-global-isel -I %p/../../include %s -o /dev/null 2>&1 | FileCheck %s
include "llvm/Target/Target.td"

//===- Boiler plate target code -===//
def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

let TargetPrefix = "mytarget" in {
def int_mytarget_nop : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;
}

def R0 : Register<"r0"> { let Namespace = "MyTarget"; }
def GPR32 : RegisterClass<"MyTarget", [i32], 32, (add R0)>;

class I<dag OOps, dag IOps, list<dag> Pat>
  : Instruction {
  let Namespace = "MyTarget";
  let OutOperandList = OOps;
  let InOperandList = IOps;
  let Pattern = Pat;
}

def complex : Operand<i32>, ComplexPattern<i32, 2, "SelectComplexPattern", []> {
  let MIOperandInfo = (ops i32imm, i32imm);
}

def gi_complex :
    GIComplexOperandMatcher<s32, "selectComplexPattern">,
    GIComplexPatternEquiv<complex>;
def complex_rr : Operand<i32>, ComplexPattern<i32, 2, "SelectComplexPatternRR", []> {
  let MIOperandInfo = (ops GPR32, GPR32);
}

def gi_complex_rr :
    GIComplexOperandMatcher<s32, "selectComplexPatternRR">,
    GIComplexPatternEquiv<complex_rr>;

def INSN : I<(outs GPR32:$dst), (ins GPR32:$src1, complex:$src2), []>;

//===- Bail out when we define a variable twice wrt complex suboperands. -===//

// CHECK: warning: Skipped pattern: Complex suboperand referenced more than once (Operand: x)
def : Pat<(add (complex_rr GPR32:$x, GPR32:$y),
               (complex_rr GPR32:$x, GPR32:$z)),
               (INSN GPR32:$z, complex:$y)>;
