// RUN: llvm-tblgen -gen-global-isel -optimize-match-table=false -warn-on-skipped-patterns -I %p/../../include -I %p/Common %s -o - < %s 2>&1 | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

// Test when the inner instruction in the output pattern has two outs

def TwoOutsInstr : I<(outs GPR32:$out1, GPR32:$out2), (ins GPR32:$src), []>;
def OtherInstr : I<(outs GPR32:$dst), (ins GPR32:$src), []>;

def : Pat<(i32 (add i32:$src, i32:$src)),
  (OtherInstr (TwoOutsInstr GPR32:$src))>;

// CHECK: warning: Skipped pattern: Dst pattern child only supported with exactly one result
