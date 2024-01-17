// RUN: llvm-tblgen -gen-global-isel -I %p/../../include -I %p/Common %s | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

// CHECK:      constexpr static uint8_t MatchTable0[] = {
// CHECK-NEXT:   GIM_Try,
// CHECK-NEXT:     GIM_CheckOpcode{{.*}}GIMT_Encode2(TargetOpcode::G_ANYEXT),
// CHECK-NEXT:     GIM_CheckType{{.*}}/*Type*/GILLT_s32,
// CHECK-NEXT:     GIM_CheckType{{.*}}/*Type*/GILLT_s8,
// CHECK-NEXT:     GIM_CheckRegBankForClass{{.*}}/*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT:     // (anyext:{{.*}}=>{{.*}}(SELECT_I4:
// CHECK:          GIR_Done,
// CHECK-NEXT:   // Label 0:
// CHECK-NEXT:   GIM_Reject,
// CHECK-NEXT:   };

def SELECT_I4 : I<(outs GPR32:$dst), (ins GPR8:$cond, GPR32:$T, GPR32:$F), []>;
def LI : I<(outs GPR32:$dst), (ins i32imm:$src), []>;

def : Pat<(i32 (anyext i8:$in)), (SELECT_I4 $in, (LI 1), (LI 0))>;
