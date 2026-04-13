// RUN: llvm-tblgen -gen-global-isel -warn-on-skipped-patterns -optimize-match-table=false -I %p/../../../include -I %p/../Common %s 2> %t | FileCheck -check-prefix=GISEL %s
// RUN: FileCheck -DFILE=%s -check-prefix=ERR %s < %t

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

// GISEL: GIM_Try
// GISEL: GIM_CheckNumOperands, /*MI*/0, /*Expected*/4,
// GISEL-NEXT: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_FCMP),
// GISEL: GIM_CheckCmpPredicate, /*MI*/0, /*Op*/1, /*Predicate*/GIMT_Encode2(CmpInst::FCMP_OEQ),
def FCMPOEQ : I<(outs GPR32:$dst), (ins FPR32Op:$src0, FPR32:$src1),
              [(set GPR32:$dst, (i32 (setcc f32:$src0, f32:$src1, SETOEQ)))]>;

// GISEL: GIM_Try
// GISEL: GIM_CheckNumOperands, /*MI*/0, /*Expected*/4,
// GISEL-NEXT: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_ICMP),
// GISEL: GIM_CheckCmpPredicate, /*MI*/0, /*Op*/1, /*Predicate*/GIMT_Encode2(CmpInst::ICMP_EQ),
def ICMPEQ : I<(outs GPR32:$dst), (ins GPR32Op:$src0, GPR32:$src1),
               [(set GPR32:$dst, (i32 (setcc i32:$src0, i32:$src1, SETEQ)))]>;

// Check there is an error if not a CondCode operand.
// ERR: [[FILE]]:[[@LINE+1]]:5: warning: Skipped pattern: Unable to handle CondCode
def FCMP_NOTCC : I<(outs GPR32:$dst), (ins FPR32Op:$src0, FPR32:$src1),
                   [(set GPR32:$dst, (i32 (setcc f32:$src0, f32:$src1, i32)))]>;
