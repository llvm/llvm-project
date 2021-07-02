; RUN: llc -O0 -mtriple arm-unknown -mattr=+vfp2,+v4t -global-isel -stop-after=irtranslator -verify-machineinstrs %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=ARM -check-prefix=LITTLE
; RUN: llc -O0 -mtriple armeb-unknown -mattr=+vfp2,+v4t -global-isel -global-isel-abort=0 -stop-after=irtranslator -verify-machineinstrs %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=ARM -check-prefix=BIG
; RUN: llc -O0 -mtriple thumb-unknown -mattr=+vfp2,+v6t2 -global-isel -stop-after=irtranslator -verify-machineinstrs %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=LITTLE -check-prefix=THUMB

declare arm_aapcscc i32* @simple_reg_params_target(i32, i32*)

define arm_aapcscc i32* @test_call_simple_reg_params(i32 *%a, i32 %b) {
; CHECK-LABEL: name: test_call_simple_reg_params
; CHECK-DAG: [[AVREG:%[0-9]+]]:_(p0) = COPY $r0
; CHECK-DAG: [[BVREG:%[0-9]+]]:_(s32) = COPY $r1
; CHECK: ADJCALLSTACKDOWN 0, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK-DAG: $r0 = COPY [[BVREG]]
; CHECK-DAG: $r1 = COPY [[AVREG]]
; ARM: BL @simple_reg_params_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r1, implicit-def $r0
; THUMB: tBL 14 /* CC::al */, $noreg, @simple_reg_params_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r1, implicit-def $r0
; CHECK: [[RVREG:%[0-9]+]]:_(p0) = COPY $r0
; CHECK: ADJCALLSTACKUP 0, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK: $r0 = COPY [[RVREG]]
; ARM: BX_RET 14 /* CC::al */, $noreg, implicit $r0
; THUMB: tBX_RET 14 /* CC::al */, $noreg, implicit $r0
entry:
  %r = notail call arm_aapcscc i32 *@simple_reg_params_target(i32 %b, i32 *%a)
  ret i32 *%r
}

declare arm_aapcscc i32* @simple_stack_params_target(i32, i32*, i32, i32*, i32, i32*)

define arm_aapcscc i32* @test_call_simple_stack_params(i32 *%a, i32 %b) {
; CHECK-LABEL: name: test_call_simple_stack_params
; CHECK-DAG: [[AVREG:%[0-9]+]]:_(p0) = COPY $r0
; CHECK-DAG: [[BVREG:%[0-9]+]]:_(s32) = COPY $r1
; CHECK: ADJCALLSTACKDOWN 8, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK-DAG: $r0 = COPY [[BVREG]]
; CHECK-DAG: $r1 = COPY [[AVREG]]
; CHECK-DAG: $r2 = COPY [[BVREG]]
; CHECK-DAG: $r3 = COPY [[AVREG]]
; CHECK: [[SP1:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[OFF1:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK: [[FI1:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP1]], [[OFF1]](s32)
; CHECK: G_STORE [[BVREG]](s32), [[FI1]](p0){{.*}}store (s32)
; CHECK: [[SP2:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[OFF2:%[0-9]+]]:_(s32) = G_CONSTANT i32 4
; CHECK: [[FI2:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP2]], [[OFF2]](s32)
; CHECK: G_STORE [[AVREG]](p0), [[FI2]](p0){{.*}}store (s32)
; ARM: BL @simple_stack_params_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r1, implicit $r2, implicit $r3, implicit-def $r0
; THUMB: tBL 14 /* CC::al */, $noreg, @simple_stack_params_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r1, implicit $r2, implicit $r3, implicit-def $r0
; CHECK: [[RVREG:%[0-9]+]]:_(p0) = COPY $r0
; CHECK: ADJCALLSTACKUP 8, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK: $r0 = COPY [[RVREG]]
; ARM: BX_RET 14 /* CC::al */, $noreg, implicit $r0
; THUMB: tBX_RET 14 /* CC::al */, $noreg, implicit $r0
entry:
  %r = notail call arm_aapcscc i32 *@simple_stack_params_target(i32 %b, i32 *%a, i32 %b, i32 *%a, i32 %b, i32 *%a)
  ret i32 *%r
}

declare arm_aapcscc signext i16 @ext_target(i8 signext, i8 zeroext, i16 signext, i16 zeroext, i8 signext, i8 zeroext, i16 signext, i16 zeroext, i1 zeroext)

define arm_aapcscc signext i16 @test_call_ext_params(i8 %a, i16 %b, i1 %c) {
; CHECK-LABEL: name: test_call_ext_params
; CHECK-DAG: [[R0VREG:%[0-9]+]]:_(s32) = COPY $r0
; CHECK-DAG: [[AVREG:%[0-9]+]]:_(s8) = G_TRUNC [[R0VREG]]
; CHECK-DAG: [[R1VREG:%[0-9]+]]:_(s32) = COPY $r1
; CHECK-DAG: [[BVREG:%[0-9]+]]:_(s16) = G_TRUNC [[R1VREG]]
; CHECK-DAG: [[R2VREG:%[0-9]+]]:_(s32) = COPY $r2
; CHECK-DAG: [[CVREG:%[0-9]+]]:_(s1) = G_TRUNC [[R2VREG]]
; CHECK: ADJCALLSTACKDOWN 20, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK: [[SEXTA:%[0-9]+]]:_(s32) = G_SEXT [[AVREG]](s8)
; CHECK: $r0 = COPY [[SEXTA]]
; CHECK: [[ZEXTA:%[0-9]+]]:_(s32) = G_ZEXT [[AVREG]](s8)
; CHECK: $r1 = COPY [[ZEXTA]]
; CHECK: [[SEXTB:%[0-9]+]]:_(s32) = G_SEXT [[BVREG]](s16)
; CHECK: $r2 = COPY [[SEXTB]]
; CHECK: [[ZEXTB:%[0-9]+]]:_(s32) = G_ZEXT [[BVREG]](s16)
; CHECK: $r3 = COPY [[ZEXTB]]
; CHECK: [[SEXTA2:%[0-9]+]]:_(s32) = G_SEXT [[AVREG]]
; CHECK: [[SP1:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[OFF1:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK: [[FI1:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP1]], [[OFF1]](s32)
; CHECK: G_STORE [[SEXTA2]](s32), [[FI1]](p0){{.*}}store (s32)
; CHECK: [[ZEXTA2:%[0-9]+]]:_(s32) = G_ZEXT [[AVREG]]
; CHECK: [[SP2:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[OFF2:%[0-9]+]]:_(s32) = G_CONSTANT i32 4
; CHECK: [[FI2:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP2]], [[OFF2]](s32)
; CHECK: G_STORE [[ZEXTA2]](s32), [[FI2]](p0){{.*}}store (s32)
; CHECK: [[SEXTB2:%[0-9]+]]:_(s32) = G_SEXT [[BVREG]]
; CHECK: [[SP3:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[OFF3:%[0-9]+]]:_(s32) = G_CONSTANT i32 8
; CHECK: [[FI3:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP3]], [[OFF3]](s32)
; CHECK: G_STORE [[SEXTB2]](s32), [[FI3]](p0){{.*}}store (s32)
; CHECK: [[ZEXTB2:%[0-9]+]]:_(s32) = G_ZEXT [[BVREG]]
; CHECK: [[SP4:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[OFF4:%[0-9]+]]:_(s32) = G_CONSTANT i32 12
; CHECK: [[FI4:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP4]], [[OFF4]](s32)
; CHECK: G_STORE [[ZEXTB2]](s32), [[FI4]](p0){{.*}}store (s32)
; CHECK: [[ZEXTC:%[0-9]+]]:_(s32) = G_ZEXT [[CVREG]]
; CHECK: [[SP5:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[OFF5:%[0-9]+]]:_(s32) = G_CONSTANT i32 16
; CHECK: [[FI5:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP5]], [[OFF5]](s32)
; CHECK: G_STORE [[ZEXTC]](s32), [[FI5]](p0){{.*}}store (s32)
; ARM: BL @ext_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r1, implicit $r2, implicit $r3, implicit-def $r0
; THUMB: tBL 14 /* CC::al */, $noreg, @ext_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r1, implicit $r2, implicit $r3, implicit-def $r0
; CHECK: [[R0VREG:%[0-9]+]]:_(s32) = COPY $r0
; CHECK: [[R0VREG_ASSERT:%[0-9]+]]:_(s32) = G_ASSERT_SEXT [[R0VREG]], 16
; CHECK: [[RVREG:%[0-9]+]]:_(s16) = G_TRUNC [[R0VREG_ASSERT]]
; CHECK: ADJCALLSTACKUP 20, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK: [[RExtVREG:%[0-9]+]]:_(s32) = G_SEXT [[RVREG]]
; CHECK: $r0 = COPY [[RExtVREG]]
; ARM: BX_RET 14 /* CC::al */, $noreg, implicit $r0
; THUMB: tBX_RET 14 /* CC::al */, $noreg, implicit $r0
entry:
  %r = notail call arm_aapcscc signext i16 @ext_target(i8 signext %a, i8 zeroext %a, i16 signext %b, i16 zeroext %b, i8 signext %a, i8 zeroext %a, i16 signext %b, i16 zeroext %b, i1 zeroext %c)
  ret i16 %r
}

declare arm_aapcs_vfpcc double @vfpcc_fp_target(float, double)

define arm_aapcs_vfpcc double @test_call_vfpcc_fp_params(double %a, float %b) {
; CHECK-LABEL: name: test_call_vfpcc_fp_params
; CHECK-DAG: [[AVREG:%[0-9]+]]:_(s64) = COPY $d0
; CHECK-DAG: [[BVREG:%[0-9]+]]:_(s32) = COPY $s2
; CHECK: ADJCALLSTACKDOWN 0, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK-DAG: $s0 = COPY [[BVREG]]
; CHECK-DAG: $d1 = COPY [[AVREG]]
; ARM: BL @vfpcc_fp_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $s0, implicit $d1, implicit-def $d0
; THUMB: tBL 14 /* CC::al */, $noreg, @vfpcc_fp_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $s0, implicit $d1, implicit-def $d0
; CHECK: [[RVREG:%[0-9]+]]:_(s64) = COPY $d0
; CHECK: ADJCALLSTACKUP 0, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK: $d0 = COPY [[RVREG]]
; ARM: BX_RET 14 /* CC::al */, $noreg, implicit $d0
; THUMB: tBX_RET 14 /* CC::al */, $noreg, implicit $d0
entry:
  %r = notail call arm_aapcs_vfpcc double @vfpcc_fp_target(float %b, double %a)
  ret double %r
}

declare arm_aapcscc double @aapcscc_fp_target(float, double, float, double)

define arm_aapcscc double @test_call_aapcs_fp_params(double %a, float %b) {
; CHECK-LABEL: name: test_call_aapcs_fp_params
; CHECK-DAG: [[A1:%[0-9]+]]:_(s32) = COPY $r0
; CHECK-DAG: [[A2:%[0-9]+]]:_(s32) = COPY $r1
; LITTLE-DAG: [[AVREG:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[A1]](s32), [[A2]](s32)
; BIG-DAG: [[AVREG:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[A2]](s32), [[A1]](s32)
; CHECK-DAG: [[BVREG:%[0-9]+]]:_(s32) = COPY $r2
; CHECK: ADJCALLSTACKDOWN 16, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK-DAG: $r0 = COPY [[BVREG]]
; CHECK-DAG: [[A1:%[0-9]+]]:_(s32), [[A2:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[AVREG]](s64)
; LITTLE-DAG: $r2 = COPY [[A1]]
; LITTLE-DAG: $r3 = COPY [[A2]]
; BIG-DAG: $r2 = COPY [[A2]]
; BIG-DAG: $r3 = COPY [[A1]]
; CHECK: [[SP1:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[OFF1:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK: [[FI1:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP1]], [[OFF1]](s32)
; CHECK: G_STORE [[BVREG]](s32), [[FI1]](p0){{.*}}store (s32)
; CHECK: [[SP2:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[OFF2:%[0-9]+]]:_(s32) = G_CONSTANT i32 8
; CHECK: [[FI2:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP2]], [[OFF2]](s32)
; CHECK: G_STORE [[AVREG]](s64), [[FI2]](p0){{.*}}store (s64)
; ARM: BL @aapcscc_fp_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r2, implicit $r3, implicit-def $r0, implicit-def $r1
; THUMB: tBL 14 /* CC::al */, $noreg, @aapcscc_fp_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r2, implicit $r3, implicit-def $r0, implicit-def $r1
; CHECK-DAG: [[R1:%[0-9]+]]:_(s32) = COPY $r0
; CHECK-DAG: [[R2:%[0-9]+]]:_(s32) = COPY $r1
; LITTLE: [[RVREG:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[R1]](s32), [[R2]](s32)
; BIG: [[RVREG:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[R2]](s32), [[R1]](s32)
; CHECK: ADJCALLSTACKUP 16, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK: [[R1:%[0-9]+]]:_(s32), [[R2:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[RVREG]](s64)
; LITTLE-DAG: $r0 = COPY [[R1]]
; LITTLE-DAG: $r1 = COPY [[R2]]
; BIG-DAG: $r0 = COPY [[R2]]
; BIG-DAG: $r1 = COPY [[R1]]
; ARM: BX_RET 14 /* CC::al */, $noreg, implicit $r0, implicit $r1
; THUMB: tBX_RET 14 /* CC::al */, $noreg, implicit $r0, implicit $r1
entry:
  %r = notail call arm_aapcscc double @aapcscc_fp_target(float %b, double %a, float %b, double %a)
  ret double %r
}

declare arm_aapcscc float @different_call_conv_target(float)

define arm_aapcs_vfpcc float @test_call_different_call_conv(float %x) {
; CHECK-LABEL: name: test_call_different_call_conv
; CHECK: [[X:%[0-9]+]]:_(s32) = COPY $s0
; CHECK: ADJCALLSTACKDOWN 0, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK: $r0 = COPY [[X]]
; ARM: BL @different_call_conv_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit-def $r0
; THUMB: tBL 14 /* CC::al */, $noreg, @different_call_conv_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit-def $r0
; CHECK: [[R:%[0-9]+]]:_(s32) = COPY $r0
; CHECK: ADJCALLSTACKUP 0, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK: $s0 = COPY [[R]]
; ARM: BX_RET 14 /* CC::al */, $noreg, implicit $s0
; THUMB: tBX_RET 14 /* CC::al */, $noreg, implicit $s0
entry:
  %r = notail call arm_aapcscc float @different_call_conv_target(float %x)
  ret float %r
}

declare arm_aapcscc [3 x i32] @tiny_int_arrays_target([2 x i32])

define arm_aapcscc [3 x i32] @test_tiny_int_arrays([2 x i32] %arr) {
; CHECK-LABEL: name: test_tiny_int_arrays
; CHECK: liveins: $r0, $r1
; CHECK: [[R0:%[0-9]+]]:_(s32) = COPY $r0
; CHECK: [[R1:%[0-9]+]]:_(s32) = COPY $r1
; CHECK: ADJCALLSTACKDOWN 0, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK: $r0 = COPY [[R0]]
; CHECK: $r1 = COPY [[R1]]
; ARM: BL @tiny_int_arrays_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r1, implicit-def $r0, implicit-def $r1
; THUMB: tBL 14 /* CC::al */, $noreg, @tiny_int_arrays_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r1, implicit-def $r0, implicit-def $r1
; CHECK: [[R0:%[0-9]+]]:_(s32) = COPY $r0
; CHECK: [[R1:%[0-9]+]]:_(s32) = COPY $r1
; CHECK: [[R2:%[0-9]+]]:_(s32) = COPY $r2
; CHECK: ADJCALLSTACKUP 0, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; FIXME: This doesn't seem correct with regard to the AAPCS docs (which say
; that composite types larger than 4 bytes should be passed through memory),
; but it's what DAGISel does. We should fix it in the common code for both.
; CHECK: $r0 = COPY [[R0]]
; CHECK: $r1 = COPY [[R1]]
; CHECK: $r2 = COPY [[R2]]
; ARM: BX_RET 14 /* CC::al */, $noreg, implicit $r0, implicit $r1, implicit $r2
; THUMB: tBX_RET 14 /* CC::al */, $noreg, implicit $r0, implicit $r1, implicit $r2
entry:
  %r = notail call arm_aapcscc [3 x i32] @tiny_int_arrays_target([2 x i32] %arr)
  ret [3 x i32] %r
}

declare arm_aapcscc void @multiple_int_arrays_target([2 x i32], [2 x i32])

define arm_aapcscc void @test_multiple_int_arrays([2 x i32] %arr0, [2 x i32] %arr1) {
; CHECK-LABEL: name: test_multiple_int_arrays
; CHECK: liveins: $r0, $r1
; CHECK: [[R0:%[0-9]+]]:_(s32) = COPY $r0
; CHECK: [[R1:%[0-9]+]]:_(s32) = COPY $r1
; CHECK: [[R2:%[0-9]+]]:_(s32) = COPY $r2
; CHECK: [[R3:%[0-9]+]]:_(s32) = COPY $r3
; CHECK: ADJCALLSTACKDOWN 0, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK: $r0 = COPY [[R0]]
; CHECK: $r1 = COPY [[R1]]
; CHECK: $r2 = COPY [[R2]]
; CHECK: $r3 = COPY [[R3]]
; ARM: BL @multiple_int_arrays_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r1, implicit $r2, implicit $r3
; THUMB: tBL 14 /* CC::al */, $noreg, @multiple_int_arrays_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r1, implicit $r2, implicit $r3
; CHECK: ADJCALLSTACKUP 0, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; ARM: BX_RET 14 /* CC::al */, $noreg
; THUMB: tBX_RET 14 /* CC::al */, $noreg
entry:
  notail call arm_aapcscc void @multiple_int_arrays_target([2 x i32] %arr0, [2 x i32] %arr1)
  ret void
}

declare arm_aapcscc void @large_int_arrays_target([20 x i32])

define arm_aapcscc void @test_large_int_arrays([20 x i32] %arr) {
; CHECK-LABEL: name: test_large_int_arrays
; CHECK: fixedStack:
; The parameters live in separate stack locations, one for each element that
; doesn't fit in the registers.
; CHECK-DAG: id: [[FIRST_STACK_ID:[0-9]+]], type: default, offset: 0, size: 4,
; CHECK-DAG: id: [[LAST_STACK_ID:[-0]+]], type: default, offset: 60, size: 4
; CHECK: liveins: $r0, $r1, $r2, $r3
; CHECK-DAG: [[R0:%[0-9]+]]:_(s32) = COPY $r0
; CHECK-DAG: [[R1:%[0-9]+]]:_(s32) = COPY $r1
; CHECK-DAG: [[R2:%[0-9]+]]:_(s32) = COPY $r2
; CHECK-DAG: [[R3:%[0-9]+]]:_(s32) = COPY $r3
; CHECK: [[FIRST_STACK_ELEMENT_FI:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[FIRST_STACK_ID]]
; CHECK: [[FIRST_STACK_ELEMENT:%[0-9]+]]:_(s32) = G_LOAD [[FIRST_STACK_ELEMENT_FI]]{{.*}}load (s32) from %fixed-stack.[[FIRST_STACK_ID]]
; CHECK: [[LAST_STACK_ELEMENT_FI:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[LAST_STACK_ID]]
; CHECK: [[LAST_STACK_ELEMENT:%[0-9]+]]:_(s32) = G_LOAD [[LAST_STACK_ELEMENT_FI]]{{.*}}load (s32) from %fixed-stack.[[LAST_STACK_ID]]
; CHECK: ADJCALLSTACKDOWN 64, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK: $r0 = COPY [[R0]]
; CHECK: $r1 = COPY [[R1]]
; CHECK: $r2 = COPY [[R2]]
; CHECK: $r3 = COPY [[R3]]
; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[OFF_FIRST_ELEMENT:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK: [[FIRST_STACK_ARG_ADDR:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP]], [[OFF_FIRST_ELEMENT]](s32)
; CHECK: G_STORE [[FIRST_STACK_ELEMENT]](s32), [[FIRST_STACK_ARG_ADDR]]{{.*}}store (s32)
; Match the second-to-last offset, so we can get the correct SP for the last element
; CHECK: G_CONSTANT i32 56
; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[OFF_LAST_ELEMENT:%[0-9]+]]:_(s32) = G_CONSTANT i32 60
; CHECK: [[LAST_STACK_ARG_ADDR:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP]], [[OFF_LAST_ELEMENT]](s32)
; CHECK: G_STORE [[LAST_STACK_ELEMENT]](s32), [[LAST_STACK_ARG_ADDR]]{{.*}}store (s32)
; ARM: BL @large_int_arrays_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r1, implicit $r2, implicit $r3
; THUMB: tBL 14 /* CC::al */, $noreg, @large_int_arrays_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r1, implicit $r2, implicit $r3
; CHECK: ADJCALLSTACKUP 64, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; ARM: BX_RET 14 /* CC::al */, $noreg
; THUMB: tBX_RET 14 /* CC::al */, $noreg
entry:
  notail call arm_aapcscc void @large_int_arrays_target([20 x i32] %arr)
  ret void
}

declare arm_aapcscc [2 x float] @fp_arrays_aapcs_target([3 x double])

define arm_aapcscc [2 x float] @test_fp_arrays_aapcs([3 x double] %arr) {
; CHECK-LABEL: name: test_fp_arrays_aapcs
; CHECK: fixedStack:
; CHECK: id: [[ARR2_ID:[0-9]+]], type: default, offset: 0, size: 8,
; CHECK: liveins: $r0, $r1, $r2, $r3
; CHECK: [[ARR0_0:%[0-9]+]]:_(s32) = COPY $r0
; CHECK: [[ARR0_1:%[0-9]+]]:_(s32) = COPY $r1
; LITTLE: [[ARR0:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[ARR0_0]](s32), [[ARR0_1]](s32)
; BIG: [[ARR0:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[ARR0_1]](s32), [[ARR0_0]](s32)
; CHECK: [[ARR1_0:%[0-9]+]]:_(s32) = COPY $r2
; CHECK: [[ARR1_1:%[0-9]+]]:_(s32) = COPY $r3
; LITTLE: [[ARR1:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[ARR1_0]](s32), [[ARR1_1]](s32)
; BIG: [[ARR1:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[ARR1_1]](s32), [[ARR1_0]](s32)
; CHECK: [[ARR2_FI:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[ARR2_ID]]
; CHECK: [[ARR2:%[0-9]+]]:_(s64) = G_LOAD [[ARR2_FI]]{{.*}}load (s64) from %fixed-stack.[[ARR2_ID]]
; CHECK: ADJCALLSTACKDOWN 8, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK: [[ARR0_0:%[0-9]+]]:_(s32), [[ARR0_1:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[ARR0]](s64)
; LITTLE: $r0 = COPY [[ARR0_0]](s32)
; LITTLE: $r1 = COPY [[ARR0_1]](s32)
; BIG: $r0 = COPY [[ARR0_1]](s32)
; BIG: $r1 = COPY [[ARR0_0]](s32)
; CHECK: [[ARR1_0:%[0-9]+]]:_(s32), [[ARR1_1:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[ARR1]](s64)
; LITTLE: $r2 = COPY [[ARR1_0]](s32)
; LITTLE: $r3 = COPY [[ARR1_1]](s32)
; BIG: $r2 = COPY [[ARR1_1]](s32)
; BIG: $r3 = COPY [[ARR1_0]](s32)
; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[ARR2_OFFSET:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK: [[ARR2_ADDR:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP]], [[ARR2_OFFSET]](s32)
; CHECK: G_STORE [[ARR2]](s64), [[ARR2_ADDR]](p0){{.*}}store (s64)
; ARM: BL @fp_arrays_aapcs_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r1, implicit $r2, implicit $r3, implicit-def $r0, implicit-def $r1
; THUMB: tBL 14 /* CC::al */, $noreg, @fp_arrays_aapcs_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r1, implicit $r2, implicit $r3, implicit-def $r0, implicit-def $r1
; CHECK: [[R0:%[0-9]+]]:_(s32) = COPY $r0
; CHECK: [[R1:%[0-9]+]]:_(s32) = COPY $r1
; CHECK: ADJCALLSTACKUP 8, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK: $r0 = COPY [[R0]]
; CHECK: $r1 = COPY [[R1]]
; ARM: BX_RET 14 /* CC::al */, $noreg, implicit $r0, implicit $r1
; THUMB: tBX_RET 14 /* CC::al */, $noreg, implicit $r0, implicit $r1
entry:
  %r = notail call arm_aapcscc [2 x float] @fp_arrays_aapcs_target([3 x double] %arr)
  ret [2 x float] %r
}

declare arm_aapcs_vfpcc [4 x float] @fp_arrays_aapcs_vfp_target([3 x double], [3 x float], [4 x double])

define arm_aapcs_vfpcc [4 x float] @test_fp_arrays_aapcs_vfp([3 x double] %x, [3 x float] %y, [4 x double] %z) {
; CHECK-LABEL: name: test_fp_arrays_aapcs_vfp
; CHECK: fixedStack:
; CHECK-DAG: id: [[Z0_ID:[0-9]+]], type: default, offset: 0, size: 8,
; CHECK-DAG: id: [[Z1_ID:[0-9]+]], type: default, offset: 8, size: 8,
; CHECK-DAG: id: [[Z2_ID:[0-9]+]], type: default, offset: 16, size: 8,
; CHECK-DAG: id: [[Z3_ID:[0-9]+]], type: default, offset: 24, size: 8,
; CHECK: liveins: $d0, $d1, $d2, $s6, $s7, $s8
; CHECK: [[X0:%[0-9]+]]:_(s64) = COPY $d0
; CHECK: [[X1:%[0-9]+]]:_(s64) = COPY $d1
; CHECK: [[X2:%[0-9]+]]:_(s64) = COPY $d2
; CHECK: [[Y0:%[0-9]+]]:_(s32) = COPY $s6
; CHECK: [[Y1:%[0-9]+]]:_(s32) = COPY $s7
; CHECK: [[Y2:%[0-9]+]]:_(s32) = COPY $s8
; CHECK: [[Z0_FI:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[Z0_ID]]
; CHECK: [[Z0:%[0-9]+]]:_(s64) = G_LOAD [[Z0_FI]]{{.*}}load (s64)
; CHECK: [[Z1_FI:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[Z1_ID]]
; CHECK: [[Z1:%[0-9]+]]:_(s64) = G_LOAD [[Z1_FI]]{{.*}}load (s64)
; CHECK: [[Z2_FI:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[Z2_ID]]
; CHECK: [[Z2:%[0-9]+]]:_(s64) = G_LOAD [[Z2_FI]]{{.*}}load (s64)
; CHECK: [[Z3_FI:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[Z3_ID]]
; CHECK: [[Z3:%[0-9]+]]:_(s64) = G_LOAD [[Z3_FI]]{{.*}}load (s64)
; CHECK: ADJCALLSTACKDOWN 32, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK: $d0 = COPY [[X0]](s64)
; CHECK: $d1 = COPY [[X1]](s64)
; CHECK: $d2 = COPY [[X2]](s64)
; CHECK: $s6 = COPY [[Y0]](s32)
; CHECK: $s7 = COPY [[Y1]](s32)
; CHECK: $s8 = COPY [[Y2]](s32)
; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[Z0_OFFSET:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK: [[Z0_ADDR:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP]], [[Z0_OFFSET]](s32)
; CHECK: G_STORE [[Z0]](s64), [[Z0_ADDR]](p0){{.*}}store (s64)
; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[Z1_OFFSET:%[0-9]+]]:_(s32) = G_CONSTANT i32 8
; CHECK: [[Z1_ADDR:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP]], [[Z1_OFFSET]](s32)
; CHECK: G_STORE [[Z1]](s64), [[Z1_ADDR]](p0){{.*}}store (s64)
; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[Z2_OFFSET:%[0-9]+]]:_(s32) = G_CONSTANT i32 16
; CHECK: [[Z2_ADDR:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP]], [[Z2_OFFSET]](s32)
; CHECK: G_STORE [[Z2]](s64), [[Z2_ADDR]](p0){{.*}}store (s64)
; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[Z3_OFFSET:%[0-9]+]]:_(s32) = G_CONSTANT i32 24
; CHECK: [[Z3_ADDR:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP]], [[Z3_OFFSET]](s32)
; CHECK: G_STORE [[Z3]](s64), [[Z3_ADDR]](p0){{.*}}store (s64)
; ARM: BL @fp_arrays_aapcs_vfp_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $d0, implicit $d1, implicit $d2, implicit $s6, implicit $s7, implicit $s8, implicit-def $s0, implicit-def $s1, implicit-def $s2, implicit-def $s3
; THUMB: tBL 14 /* CC::al */, $noreg, @fp_arrays_aapcs_vfp_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $d0, implicit $d1, implicit $d2, implicit $s6, implicit $s7, implicit $s8, implicit-def $s0, implicit-def $s1, implicit-def $s2, implicit-def $s3
; CHECK: [[R0:%[0-9]+]]:_(s32) = COPY $s0
; CHECK: [[R1:%[0-9]+]]:_(s32) = COPY $s1
; CHECK: [[R2:%[0-9]+]]:_(s32) = COPY $s2
; CHECK: [[R3:%[0-9]+]]:_(s32) = COPY $s3
; CHECK: ADJCALLSTACKUP 32, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK: $s0 = COPY [[R0]]
; CHECK: $s1 = COPY [[R1]]
; CHECK: $s2 = COPY [[R2]]
; CHECK: $s3 = COPY [[R3]]
; ARM: BX_RET 14 /* CC::al */, $noreg, implicit $s0, implicit $s1, implicit $s2, implicit $s3
; THUMB: tBX_RET 14 /* CC::al */, $noreg, implicit $s0, implicit $s1, implicit $s2, implicit $s3
entry:
  %r = notail call arm_aapcs_vfpcc [4 x float] @fp_arrays_aapcs_vfp_target([3 x double] %x, [3 x float] %y, [4 x double] %z)
  ret [4 x float] %r
}

declare arm_aapcscc [2 x i32*] @tough_arrays_target([6 x [4 x i32]] %arr)

define arm_aapcscc [2 x i32*] @test_tough_arrays([6 x [4 x i32]] %arr) {
; CHECK-LABEL: name: test_tough_arrays
; CHECK: fixedStack:
; The parameters live in separate stack locations, one for each element that
; doesn't fit in the registers.
; CHECK-DAG: id: [[FIRST_STACK_ID:[0-9]+]], type: default, offset: 0, size: 4,
; CHECK-DAG: id: [[LAST_STACK_ID:[-0]+]], type: default, offset: 76, size: 4
; CHECK: liveins: $r0, $r1, $r2, $r3
; CHECK-DAG: [[R0:%[0-9]+]]:_(s32) = COPY $r0
; CHECK-DAG: [[R1:%[0-9]+]]:_(s32) = COPY $r1
; CHECK-DAG: [[R2:%[0-9]+]]:_(s32) = COPY $r2
; CHECK-DAG: [[R3:%[0-9]+]]:_(s32) = COPY $r3
; CHECK: [[FIRST_STACK_ELEMENT_FI:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[FIRST_STACK_ID]]
; CHECK: [[FIRST_STACK_ELEMENT:%[0-9]+]]:_(s32) = G_LOAD [[FIRST_STACK_ELEMENT_FI]]{{.*}}load (s32) from %fixed-stack.[[FIRST_STACK_ID]]
; CHECK: [[LAST_STACK_ELEMENT_FI:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[LAST_STACK_ID]]
; CHECK: [[LAST_STACK_ELEMENT:%[0-9]+]]:_(s32) = G_LOAD [[LAST_STACK_ELEMENT_FI]]{{.*}}load (s32) from %fixed-stack.[[LAST_STACK_ID]]
; CHECK: ADJCALLSTACKDOWN 80, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK: $r0 = COPY [[R0]]
; CHECK: $r1 = COPY [[R1]]
; CHECK: $r2 = COPY [[R2]]
; CHECK: $r3 = COPY [[R3]]
; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[OFF_FIRST_ELEMENT:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK: [[FIRST_STACK_ARG_ADDR:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP]], [[OFF_FIRST_ELEMENT]](s32)
; CHECK: G_STORE [[FIRST_STACK_ELEMENT]](s32), [[FIRST_STACK_ARG_ADDR]]{{.*}}store (s32)
; Match the second-to-last offset, so we can get the correct SP for the last element
; CHECK: G_CONSTANT i32 72
; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[OFF_LAST_ELEMENT:%[0-9]+]]:_(s32) = G_CONSTANT i32 76
; CHECK: [[LAST_STACK_ARG_ADDR:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP]], [[OFF_LAST_ELEMENT]](s32)
; CHECK: G_STORE [[LAST_STACK_ELEMENT]](s32), [[LAST_STACK_ARG_ADDR]]{{.*}}store (s32)
; ARM: BL @tough_arrays_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r1, implicit $r2, implicit $r3, implicit-def $r0, implicit-def $r1
; THUMB: tBL 14 /* CC::al */, $noreg, @tough_arrays_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r1, implicit $r2, implicit $r3, implicit-def $r0, implicit-def $r1
; CHECK: [[R0:%[0-9]+]]:_(p0) = COPY $r0
; CHECK: [[R1:%[0-9]+]]:_(p0) = COPY $r1
; CHECK: ADJCALLSTACKUP 80, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK: $r0 = COPY [[R0]]
; CHECK: $r1 = COPY [[R1]]
; ARM: BX_RET 14 /* CC::al */, $noreg, implicit $r0, implicit $r1
; THUMB: tBX_RET 14 /* CC::al */, $noreg, implicit $r0, implicit $r1
entry:
  %r = notail call arm_aapcscc [2 x i32*] @tough_arrays_target([6 x [4 x i32]] %arr)
  ret [2 x i32*] %r
}

declare arm_aapcscc {i32, i32} @structs_target({i32, i32})

define arm_aapcscc {i32, i32} @test_structs({i32, i32} %x) {
; CHECK-LABEL: test_structs
; CHECK: liveins: $r0, $r1
; CHECK-DAG: [[X0:%[0-9]+]]:_(s32) = COPY $r0
; CHECK-DAG: [[X1:%[0-9]+]]:_(s32) = COPY $r1
; CHECK: ADJCALLSTACKDOWN 0, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK-DAG: $r0 = COPY [[X0]](s32)
; CHECK-DAG: $r1 = COPY [[X1]](s32)
; ARM: BL @structs_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r1, implicit-def $r0, implicit-def $r1
; THUMB: tBL 14 /* CC::al */, $noreg, @structs_target, csr_aapcs, implicit-def $lr, implicit $sp, implicit $r0, implicit $r1, implicit-def $r0, implicit-def $r1
; CHECK: [[R0:%[0-9]+]]:_(s32) = COPY $r0
; CHECK: [[R1:%[0-9]+]]:_(s32) = COPY $r1
; CHECK: ADJCALLSTACKUP 0, 0, 14 /* CC::al */, $noreg, implicit-def $sp, implicit $sp
; CHECK: $r0 = COPY [[R0]](s32)
; CHECK: $r1 = COPY [[R1]](s32)
; ARM: BX_RET 14 /* CC::al */, $noreg, implicit $r0, implicit $r1
; THUMB: tBX_RET 14 /* CC::al */, $noreg, implicit $r0, implicit $r1
  %r = notail call arm_aapcscc {i32, i32} @structs_target({i32, i32} %x)
  ret {i32, i32} %r
}
