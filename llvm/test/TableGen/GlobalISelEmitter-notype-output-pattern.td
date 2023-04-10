// RUN: llvm-tblgen -gen-global-isel -I %p/../../include -I %p/Common %s | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

// No rule will be added to the match table.
// CHECK:      constexpr static int64_t MatchTable0[] = {
// CHECK-NEXT:   GIM_Reject
// CHECK-NEXT: };

def SELECT_I4 : I<(outs GPR32:$dst), (ins GPR8:$cond, GPR32:$T, GPR32:$F), []>;
def LI : I<(outs GPR32:$dst), (ins i32imm:$src), []>;

def : Pat<(i32 (anyext i8:$in)), (SELECT_I4 $in, (LI 1), (LI 0))>;
