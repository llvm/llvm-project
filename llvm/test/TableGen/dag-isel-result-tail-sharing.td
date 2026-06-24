// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include %s -o - \
// RUN:   | FileCheck --check-prefix=SHARED %s
// RUN: llvm-tblgen -gen-dag-isel -DSTRUCTURAL_DIFFERENCE \
// RUN:   -I %p/../../include %s -o - | FileCheck --check-prefix=DIFFERENT %s
// RUN: llvm-tblgen -gen-dag-isel -DFAILURE_PREFIX \
// RUN:   -I %p/../../include %s -o - | FileCheck --check-prefix=FAILURE %s
// RUN: llvm-tblgen -gen-dag-isel -instrument-coverage \
// RUN:   -I %p/../../include %s -o - | FileCheck --check-prefix=COVERAGE %s

include "llvm/Target/Target.td"

def TestInstrInfo : InstrInfo;
def TestTarget : Target {
  let InstructionSet = TestInstrInfo;
}

def R0 : Register<"r0">;
def GPR : RegisterClass<"TestTarget", [i32], 32, (add R0)>;

def WRAP : Instruction {
  let OutOperandList = (outs GPR:$dst);
  let InOperandList = (ins GPR:$src);
}

def WRAP_ALT : Instruction {
  let OutOperandList = (outs GPR:$dst);
  let InOperandList = (ins GPR:$src);
}

def PAIR : Instruction {
  let OutOperandList = (outs GPR:$dst);
  let InOperandList = (ins GPR:$lhs, GPR:$rhs);
}

#ifndef STRUCTURAL_DIFFERENCE
#ifndef FAILURE_PREFIX
// The ADD and SUB patterns have different predicates but identical generated
// results. Their result suffix is long enough to share.
def : Pat<
    (add GPR:$x, GPR:$y),
    (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP
        (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP
            (PAIR GPR:$x, GPR:$y)))))))))))))))))>;

def : Pat<
    (sub GPR:$x, GPR:$y),
    (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP
        (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP
            (PAIR GPR:$x, GPR:$y)))))))))))))))))>;

// SHARED: OPC_Jump, JUMP_TARGET([[TAIL:[0-9]+]]),
// SHARED: OPC_Jump, JUMP_TARGET([[TAIL]]),
// SHARED: // Shared result tail at [[TAIL]]
// SHARED: OPC_EmitNode
// SHARED: OPC_MorphNodeTo
#endif
#endif

#ifdef STRUCTURAL_DIFFERENCE
// These result sequences have the same matcher kinds and encoded sizes, but
// different instruction records. The emitted matcher instructions must keep them separate.
def : Pat<
    (add GPR:$x, GPR:$y),
    (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP
        (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP
            (PAIR GPR:$x, GPR:$y)))))))))))))))))>;

def : Pat<
    (sub GPR:$x, GPR:$y),
    (WRAP_ALT (WRAP_ALT (WRAP_ALT (WRAP_ALT (WRAP_ALT (WRAP_ALT
        (WRAP_ALT (WRAP_ALT (WRAP_ALT (WRAP_ALT (WRAP_ALT (WRAP_ALT
        (WRAP_ALT (WRAP_ALT (WRAP_ALT (WRAP_ALT
            (PAIR GPR:$x, GPR:$y)))))))))))))))))>;

// DIFFERENT-NOT: OPC_Jump
#endif

#ifdef FAILURE_PREFIX
// EmitMergeInputChains may fail after inspecting the matched DAG. Each pattern
// must execute its own merge before entering the shared non-failing suffix.
def int_chain_a : Intrinsic<
    [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty]>;
def int_chain_b : Intrinsic<
    [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty]>;

def : Pat<
    (int_chain_a GPR:$x, GPR:$y),
    (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP
        (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP
            (PAIR GPR:$x, GPR:$y)))))))))))))))))>;

def : Pat<
    (int_chain_b GPR:$x, GPR:$y),
    (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP
        (WRAP (WRAP (WRAP (WRAP (WRAP (WRAP
            (PAIR GPR:$x, GPR:$y)))))))))))))))))>;

// FAILURE: OPC_EmitMergeInputChains
// FAILURE-NEXT: {{.*}}OPC_Jump, JUMP_TARGET([[TAIL:[0-9]+]]),
// FAILURE: OPC_EmitMergeInputChains
// FAILURE-NEXT: {{.*}}OPC_Jump, JUMP_TARGET([[TAIL]]),
// FAILURE: // Shared result tail at [[TAIL]]
// FAILURE-NOT: OPC_EmitMergeInputChains
// FAILURE: OPC_EmitNode
#endif

// Coverage output must preserve one coverage record per source pattern.
// COVERAGE-NOT: OPC_Jump
// COVERAGE-COUNT-2: OPC_Coverage
