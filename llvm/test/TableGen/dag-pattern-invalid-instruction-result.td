// RUN: not llvm-tblgen -gen-instr-info -I %p/../../include %s -o /dev/null 2>&1 | FileCheck %s
// RUN: llvm-tblgen -gen-instr-info -I %p/../../include %s -DVALID_RESULT -o /dev/null
// RUN: not llvm-tblgen -gen-dag-isel -I %p/../../include %s -o /dev/null 2>&1 | FileCheck %s
// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include %s -DVALID_RESULT -o /dev/null

include "llvm/Target/Target.td"

def : Target { let InstructionSet = InstrInfo<>; }
def GPR : RegisterClass<"Test", [i32], 32, (add Register<"r0">)>;
defm : RemapAllTargetPseudoPointerOperands<GPR>;

// CHECK: :[[@LINE+1]]:5: error: In Bad: Operand $dst does not exist in operand list!
def Bad : Instruction {
  let Namespace = "Test";
  let OutOperandList = (outs GPR:$dst);
  let InOperandList = (ins);
#ifdef VALID_RESULT
  let Pattern = [(set GPR:$dst, 0)];
#else
  let Pattern = [(set GPR:$other, 0)];
#endif
}
