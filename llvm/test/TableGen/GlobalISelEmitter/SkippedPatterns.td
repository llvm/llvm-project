// RUN: llvm-tblgen -warn-on-skipped-patterns -gen-global-isel -I %p/../../../include %s -I %p/../Common -o /dev/null 2>&1 | FileCheck %s
// RUN: llvm-tblgen -warn-on-skipped-patterns -gen-global-isel -I %p/../../../include %s -I %p/../Common -o /dev/null -DIGNORE 2>&1 | FileCheck --allow-empty --check-prefix=IGNORED %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

//===- Boiler plate target code -===//

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

#ifdef IGNORE
let GISelShouldIgnore = 1 in
#endif
// IGNORED-NOT: warning: Skipped pattern: Error: {{.*}}
// CHECK: warning: Skipped pattern: Error: Complex suboperand x referenced by different operands: complex_rr:x:y and complex_rr:x:z.
def : Pat<(add (complex_rr GPR32:$x, GPR32:$y),
               (complex_rr GPR32:$x, GPR32:$z)),
               (INSN GPR32:$z, complex:$y)>;
