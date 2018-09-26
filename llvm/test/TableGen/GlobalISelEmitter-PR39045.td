// RUN: llvm-tblgen -gen-global-isel -I %p/../../include %s -o %t
// RUN: FileCheck %s < %t

// Both predicates should be tested
// CHECK-DAG: GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIPFP_MI_Predicate_pat_frag_b,
// CHECK-DAG: GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIPFP_MI_Predicate_pat_frag_a,

include "llvm/Target/Target.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }


def pat_frag_a : PatFrag <(ops node:$ptr), (load node:$ptr), [{}]> {
   let PredicateCode = [{ return isInstA(MI); }];
   let GISelPredicateCode = [{ return isInstA(MI); }];
}

def pat_frag_b : PatFrag <(ops node:$ptr), (load node:$ptr), [{}]> {
   let PredicateCode = [{ return isInstB(MI); }];
   let GISelPredicateCode = [{ return isInstB(MI); }];
}

def R0 : Register<"r0"> { let Namespace = "MyTarget"; }
def GPR32 : RegisterClass<"MyTarget", [i32], 32, (add R0)>;

def inst_a : Instruction {
  let OutOperandList = (outs GPR32:$dst);
  let InOperandList = (ins GPR32:$src);
}
def inst_b : Instruction {
  let OutOperandList = (outs GPR32:$dst);
  let InOperandList = (ins GPR32:$src);
}

def : Pat <
  (pat_frag_a GPR32:$src),
  (inst_a GPR32:$src)
>;

def : Pat <
  (pat_frag_b GPR32:$src),
  (inst_b GPR32:$src)
>;

