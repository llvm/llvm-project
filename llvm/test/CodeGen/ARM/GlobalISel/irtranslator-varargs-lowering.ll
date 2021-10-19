; RUN: llc -mtriple arm-unknown -mattr=+vfp2,+v6t2 -global-isel -stop-after=irtranslator -verify-machineinstrs %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=ARM
; RUN: llc -mtriple thumb-unknown -mattr=+vfp2,+v6t2 -global-isel -stop-after=irtranslator -verify-machineinstrs %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=THUMB

declare arm_aapcscc i32 @int_varargs_target(i32, ...)

define arm_aapcscc i32 @test_call_to_varargs_with_ints(i32 *%a, i32 %b) {
; CHECK-LABEL: name: test_call_to_varargs_with_ints
; CHECK-DAG: [[AVREG:%[0-9]+]]:_(p0) = COPY $r0
; CHECK-DAG: [[BVREG:%[0-9]+]]:_(s32) = COPY $r1
; CHECK: ADJCALLSTACKDOWN 8, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK: [[SP1:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[OFF1:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK: [[FI1:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP1]], [[OFF1]](s32)
; CHECK: G_STORE [[BVREG]](s32), [[FI1]](p0){{.*}}store (s32)
; CHECK: [[SP2:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[OFF2:%[0-9]+]]:_(s32) = G_CONSTANT i32 4
; CHECK: [[FI2:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP2]], [[OFF2]](s32)
; CHECK: G_STORE [[AVREG]](p0), [[FI2]](p0){{.*}}store (p0)
; CHECK-DAG: $r0 = COPY [[BVREG]]
; CHECK-DAG: $r1 = COPY [[AVREG]]
; CHECK-DAG: $r2 = COPY [[BVREG]]
; CHECK-DAG: $r3 = COPY [[AVREG]]
; ARM: BL @int_varargs_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r1, implicit $r2, implicit $r3, implicit-def $r0
; THUMB: tBL 14 /* CC::al */, $noreg, @int_varargs_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r1, implicit $r2, implicit $r3, implicit-def $r0
; CHECK: [[RVREG:%[0-9]+]]:_(s32) = COPY $r0
; CHECK: ADJCALLSTACKUP 8, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK: $r0 = COPY [[RVREG]]
; ARM: BX_RET 14 /* CC::al */, $noreg, implicit $r0
; THUMB: tBX_RET 14 /* CC::al */, $noreg, implicit $r0
entry:
  %r = notail call arm_aapcscc i32(i32, ...) @int_varargs_target(i32 %b, i32 *%a, i32 %b, i32 *%a, i32 %b, i32 *%a)
  ret i32 %r
}

declare arm_aapcs_vfpcc float @float_varargs_target(float, double, ...)

define arm_aapcs_vfpcc float @test_call_to_varargs_with_floats(float %a, double %b) {
; CHECK-LABEL: name: test_call_to_varargs_with_floats
; CHECK-DAG: [[AVREG:%[0-9]+]]:_(s32) = COPY $s0
; CHECK-DAG: [[BVREG:%[0-9]+]]:_(s64) = COPY $d1
; CHECK: ADJCALLSTACKDOWN 8, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK-DAG: [[B1:%[0-9]+]]:_(s32), [[B2:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[BVREG]](s64)
; CHECK: [[SP1:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[OFF1:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK: [[FI1:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP1]], [[OFF1]](s32)
; CHECK: G_STORE [[BVREG]](s64), [[FI1]](p0){{.*}}store (s64)
; CHECK-DAG: $r0 = COPY [[AVREG]]
; CHECK-DAG: $r2 = COPY [[B1]]
; CHECK-DAG: $r3 = COPY [[B2]]
; ARM: BL @float_varargs_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r2, implicit $r3, implicit-def $r0
; THUMB: tBL 14 /* CC::al */, $noreg, @float_varargs_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r2, implicit $r3, implicit-def $r0
; CHECK: [[RVREG:%[0-9]+]]:_(s32) = COPY $r0
; CHECK: ADJCALLSTACKUP 8, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK: $s0 = COPY [[RVREG]]
; ARM: BX_RET 14 /* CC::al */, $noreg, implicit $s0
; THUMB: tBX_RET 14 /* CC::al */, $noreg, implicit $s0
entry:
  %r = notail call arm_aapcs_vfpcc float(float, double, ...) @float_varargs_target(float %a, double %b, double %b)
  ret float %r
}

define arm_aapcs_vfpcc float @test_call_to_varargs_with_floats_fixed_args_only(float %a, double %b) {
; CHECK-LABEL: name: test_call_to_varargs_with_floats_fixed_args_only
; CHECK-DAG: [[AVREG:%[0-9]+]]:_(s32) = COPY $s0
; CHECK-DAG: [[BVREG:%[0-9]+]]:_(s64) = COPY $d1
; CHECK: ADJCALLSTACKDOWN 0, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK-DAG: $r0 = COPY [[AVREG]]
; CHECK-DAG: [[B1:%[0-9]+]]:_(s32), [[B2:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[BVREG]](s64)
; CHECK-DAG: $r2 = COPY [[B1]]
; CHECK-DAG: $r3 = COPY [[B2]]
; ARM: BL @float_varargs_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r2, implicit $r3, implicit-def $r0
; THUMB: tBL 14 /* CC::al */, $noreg, @float_varargs_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r2, implicit $r3, implicit-def $r0
; CHECK: [[RVREG:%[0-9]+]]:_(s32) = COPY $r0
; CHECK: ADJCALLSTACKUP 0, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK: $s0 = COPY [[RVREG]]
; ARM: BX_RET 14 /* CC::al */, $noreg, implicit $s0
; THUMB: tBX_RET 14 /* CC::al */, $noreg, implicit $s0
entry:
  %r = notail call arm_aapcs_vfpcc float(float, double, ...) @float_varargs_target(float %a, double %b)
  ret float %r
}

define arm_aapcs_vfpcc float @test_indirect_call_to_varargs(float (float, double, ...) *%fptr, float %a, double %b) {
; CHECK-LABEL: name: test_indirect_call_to_varargs
; CHECK-DAG: [[FPTRVREG:%[0-9]+]]:gpr(p0) = COPY $r0
; CHECK-DAG: [[AVREG:%[0-9]+]]:_(s32) = COPY $s0
; CHECK-DAG: [[BVREG:%[0-9]+]]:_(s64) = COPY $d1
; CHECK: ADJCALLSTACKDOWN 8, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK-DAG: [[B1:%[0-9]+]]:_(s32), [[B2:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[BVREG]](s64)
; CHECK: [[SP1:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[OFF1:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK: [[FI1:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP1]], [[OFF1]](s32)
; CHECK: G_STORE [[BVREG]](s64), [[FI1]](p0){{.*}}store (s64)
; CHECK-DAG: $r0 = COPY [[AVREG]]
; CHECK-DAG: $r2 = COPY [[B1]]
; CHECK-DAG: $r3 = COPY [[B2]]
; ARM: BLX [[FPTRVREG]](p0), csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r2, implicit $r3, implicit-def $r0
; THUMB: tBLXr 14 /* CC::al */, $noreg, [[FPTRVREG]](p0), csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r2, implicit $r3, implicit-def $r0
; CHECK: [[RVREG:%[0-9]+]]:_(s32) = COPY $r0
; CHECK: ADJCALLSTACKUP 8, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK: $s0 = COPY [[RVREG]]
; ARM: BX_RET 14 /* CC::al */, $noreg, implicit $s0
; THUMB: tBX_RET 14 /* CC::al */, $noreg, implicit $s0
entry:
  %r = notail call arm_aapcs_vfpcc float(float, double, ...) %fptr(float %a, double %b, double %b)
  ret float %r
}
