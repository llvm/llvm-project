// RUN: not llvm-tblgen -gen-dag-isel -I %p/../../include -I %p/Common -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// RUN: not llvm-tblgen -gen-dag-isel -I %p/../../include -I %p/Common -DERROR2 %s 2>&1 | FileCheck --check-prefix=ERROR2 %s
// RUN: not llvm-tblgen -gen-dag-isel -I %p/../../include -I %p/Common -DERROR3 %s 2>&1 | FileCheck --check-prefix=ERROR3 %s
// RUN: not llvm-tblgen -gen-dag-isel -I %p/../../include -I %p/Common -DERROR4 %s 2>&1 | FileCheck --check-prefix=ERROR4 %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

def int_foo : Intrinsic<[llvm_anyint_ty, llvm_anyint_ty], [llvm_i32_ty]>;
def int_bar : Intrinsic<[], []>;

def INSTR_FOO : Instruction {
  let OutOperandList = (outs GPR32:$a, GPR32:$b);
  let InOperandList = (ins GPR32:$c);
}
def INSTR_BAR : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins);
}

#ifdef ERROR1
// ERROR1: [[@LINE+1]]:1: error: {{.*}} Invalid number of type casts!
def : Pat<([i32, i32, i32] (int_foo (i32 GPR32:$a))), ([i32, i32, i32] (INSTR_FOO $a))>;
#endif

#ifdef ERROR2
// ERROR2: [[@LINE+1]]:1: error: {{.*}} Invalid number of type casts!
def : Pat<([]<ValueType> (int_bar)), ([]<ValueType> (INSTR_BAR))>;
#endif

#ifdef ERROR3
// ERROR3: [[@LINE+1]]:1: error: {{.*}} Type cast only takes one operand!
def : Pat<([i32, i32] (int_foo), (int_foo)), ([i32, i32] (INSTR_FOO))>;
#endif

#ifdef ERROR4
// ERROR4: [[@LINE+1]]:1: error: {{.*}} Type cast should not have a name!
def : Pat<([i32, i32] ([i32, i32] (int_foo)):$name), ([i32, i32] (INSTR_FOO))>;
#endif
