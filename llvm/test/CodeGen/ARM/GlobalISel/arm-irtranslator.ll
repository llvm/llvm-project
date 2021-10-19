; RUN: llc -mtriple arm-unknown -mattr=+vfp2,+v4t -global-isel -stop-after=irtranslator -verify-machineinstrs %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=LITTLE
; RUN: llc -mtriple armeb-unknown -mattr=+vfp2,+v4t -global-isel -global-isel-abort=0 -stop-after=irtranslator -verify-machineinstrs %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=BIG

define void @test_void_return() {
; CHECK-LABEL: name: test_void_return
; CHECK: BX_RET 14 /* CC::al */, $noreg
entry:
  ret void
}

define signext i1 @test_add_i1(i1 %x, i1 %y) {
; CHECK-LABEL: name: test_add_i1
; CHECK: liveins: $r0, $r1
; CHECK-DAG: [[VREGR0:%[0-9]+]]:_(s32) = COPY $r0
; CHECK-DAG: [[VREGX:%[0-9]+]]:_(s1) = G_TRUNC [[VREGR0]]
; CHECK-DAG: [[VREGR1:%[0-9]+]]:_(s32) = COPY $r1
; CHECK-DAG: [[VREGY:%[0-9]+]]:_(s1) = G_TRUNC [[VREGR1]]
; CHECK: [[SUM:%[0-9]+]]:_(s1) = G_ADD [[VREGX]], [[VREGY]]
; CHECK: [[EXT:%[0-9]+]]:_(s32) = G_SEXT [[SUM]]
; CHECK: $r0 = COPY [[EXT]](s32)
; CHECK: BX_RET 14 /* CC::al */, $noreg, implicit $r0
entry:
  %sum = add i1 %x, %y
  ret i1 %sum
}

define i8 @test_add_i8(i8 %x, i8 %y) {
; CHECK-LABEL: name: test_add_i8
; CHECK: liveins: $r0, $r1
; CHECK-DAG: [[VREGR0:%[0-9]+]]:_(s32) = COPY $r0
; CHECK-DAG: [[VREGX:%[0-9]+]]:_(s8) = G_TRUNC [[VREGR0]]
; CHECK-DAG: [[VREGR1:%[0-9]+]]:_(s32) = COPY $r1
; CHECK-DAG: [[VREGY:%[0-9]+]]:_(s8) = G_TRUNC [[VREGR1]]
; CHECK: [[SUM:%[0-9]+]]:_(s8) = G_ADD [[VREGX]], [[VREGY]]
; CHECK: [[SUM_EXT:%[0-9]+]]:_(s32) = G_ANYEXT [[SUM]]
; CHECK: $r0 = COPY [[SUM_EXT]](s32)
; CHECK: BX_RET 14 /* CC::al */, $noreg, implicit $r0
entry:
  %sum = add i8 %x, %y
  ret i8 %sum
}

define i8 @test_sub_i8(i8 %x, i8 %y) {
; CHECK-LABEL: name: test_sub_i8
; CHECK: liveins: $r0, $r1
; CHECK-DAG: [[VREGR0:%[0-9]+]]:_(s32) = COPY $r0
; CHECK-DAG: [[VREGX:%[0-9]+]]:_(s8) = G_TRUNC [[VREGR0]]
; CHECK-DAG: [[VREGR1:%[0-9]+]]:_(s32) = COPY $r1
; CHECK-DAG: [[VREGY:%[0-9]+]]:_(s8) = G_TRUNC [[VREGR1]]
; CHECK: [[RES:%[0-9]+]]:_(s8) = G_SUB [[VREGX]], [[VREGY]]
; CHECK: [[RES_EXT:%[0-9]+]]:_(s32) = G_ANYEXT [[RES]]
; CHECK: $r0 = COPY [[RES_EXT]](s32)
; CHECK: BX_RET 14 /* CC::al */, $noreg, implicit $r0
entry:
  %res = sub i8 %x, %y
  ret i8 %res
}

define signext i8 @test_return_sext_i8(i8 %x) {
; CHECK-LABEL: name: test_return_sext_i8
; CHECK: liveins: $r0
; CHECK: [[VREGR0:%[0-9]+]]:_(s32) = COPY $r0
; CHECK: [[VREG:%[0-9]+]]:_(s8) = G_TRUNC [[VREGR0]]
; CHECK: [[VREGEXT:%[0-9]+]]:_(s32) = G_SEXT [[VREG]]
; CHECK: $r0 = COPY [[VREGEXT]](s32)
; CHECK: BX_RET 14 /* CC::al */, $noreg, implicit $r0
entry:
  ret i8 %x
}

define i16 @test_add_i16(i16 %x, i16 %y) {
; CHECK-LABEL: name: test_add_i16
; CHECK: liveins: $r0, $r1
; CHECK-DAG: [[VREGR0:%[0-9]+]]:_(s32) = COPY $r0
; CHECK-DAG: [[VREGX:%[0-9]+]]:_(s16) = G_TRUNC [[VREGR0]]
; CHECK-DAG: [[VREGR1:%[0-9]+]]:_(s32) = COPY $r1
; CHECK-DAG: [[VREGY:%[0-9]+]]:_(s16) = G_TRUNC [[VREGR1]]
; CHECK: [[SUM:%[0-9]+]]:_(s16) = G_ADD [[VREGX]], [[VREGY]]
; CHECK: [[SUM_EXT:%[0-9]+]]:_(s32) = G_ANYEXT [[SUM]]
; CHECK: $r0 = COPY [[SUM_EXT]](s32)
; CHECK: BX_RET 14 /* CC::al */, $noreg, implicit $r0
entry:
  %sum = add i16 %x, %y
  ret i16 %sum
}

define i16 @test_sub_i16(i16 %x, i16 %y) {
; CHECK-LABEL: name: test_sub_i16
; CHECK: liveins: $r0, $r1
; CHECK-DAG: [[VREGR0:%[0-9]+]]:_(s32) = COPY $r0
; CHECK-DAG: [[VREGX:%[0-9]+]]:_(s16) = G_TRUNC [[VREGR0]]
; CHECK-DAG: [[VREGR1:%[0-9]+]]:_(s32) = COPY $r1
; CHECK-DAG: [[VREGY:%[0-9]+]]:_(s16) = G_TRUNC [[VREGR1]]
; CHECK: [[RES:%[0-9]+]]:_(s16) = G_SUB [[VREGX]], [[VREGY]]
; CHECK: [[RES_EXT:%[0-9]+]]:_(s32) = G_ANYEXT [[RES]]
; CHECK: $r0 = COPY [[RES_EXT]](s32)
; CHECK: BX_RET 14 /* CC::al */, $noreg, implicit $r0
entry:
  %res = sub i16 %x, %y
  ret i16 %res
}

define zeroext i16 @test_return_zext_i16(i16 %x) {
; CHECK-LABEL: name: test_return_zext_i16
; CHECK: liveins: $r0
; CHECK: [[VREGR0:%[0-9]+]]:_(s32) = COPY $r0
; CHECK: [[VREG:%[0-9]+]]:_(s16) = G_TRUNC [[VREGR0]]
; CHECK: [[VREGEXT:%[0-9]+]]:_(s32) = G_ZEXT [[VREG]]
; CHECK: $r0 = COPY [[VREGEXT]](s32)
; CHECK: BX_RET 14 /* CC::al */, $noreg, implicit $r0
entry:
  ret i16 %x
}

define i32 @test_add_i32(i32 %x, i32 %y) {
; CHECK-LABEL: name: test_add_i32
; CHECK: liveins: $r0, $r1
; CHECK-DAG: [[VREGX:%[0-9]+]]:_(s32) = COPY $r0
; CHECK-DAG: [[VREGY:%[0-9]+]]:_(s32) = COPY $r1
; CHECK: [[SUM:%[0-9]+]]:_(s32) = G_ADD [[VREGX]], [[VREGY]]
; CHECK: $r0 = COPY [[SUM]](s32)
; CHECK: BX_RET 14 /* CC::al */, $noreg, implicit $r0
entry:
  %sum = add i32 %x, %y
  ret i32 %sum
}

define i32 @test_sub_i32(i32 %x, i32 %y) {
; CHECK-LABEL: name: test_sub_i32
; CHECK: liveins: $r0, $r1
; CHECK-DAG: [[VREGX:%[0-9]+]]:_(s32) = COPY $r0
; CHECK-DAG: [[VREGY:%[0-9]+]]:_(s32) = COPY $r1
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_SUB [[VREGX]], [[VREGY]]
; CHECK: $r0 = COPY [[RES]](s32)
; CHECK: BX_RET 14 /* CC::al */, $noreg, implicit $r0
entry:
  %res = sub i32 %x, %y
  ret i32 %res
}

define i32 @test_stack_args(i32 %p0, i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5) {
; CHECK-LABEL: name: test_stack_args
; CHECK: fixedStack:
; CHECK-DAG: id: [[P4:[0-9]]]{{.*}}offset: 0{{.*}}size: 4
; CHECK-DAG: id: [[P5:[0-9]]]{{.*}}offset: 4{{.*}}size: 4
; CHECK: liveins: $r0, $r1, $r2, $r3
; CHECK: [[VREGP2:%[0-9]+]]:_(s32) = COPY $r2
; CHECK: [[FIP5:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[P5]]
; CHECK: [[VREGP5:%[0-9]+]]:_(s32) = G_LOAD [[FIP5]]{{.*}}load (s32)
; CHECK: [[SUM:%[0-9]+]]:_(s32) = G_ADD [[VREGP2]], [[VREGP5]]
; CHECK: $r0 = COPY [[SUM]]
; CHECK: BX_RET 14 /* CC::al */, $noreg, implicit $r0
entry:
  %sum = add i32 %p2, %p5
  ret i32 %sum
}

define i16 @test_stack_args_signext(i32 %p0, i16 %p1, i8 %p2, i1 %p3,
                                    i8 signext %p4, i16 signext %p5) {
; CHECK-LABEL: name: test_stack_args_signext
; CHECK: fixedStack:
; CHECK-DAG: id: [[P4:[0-9]]]{{.*}}offset: 0{{.*}}size: 4
; CHECK-DAG: id: [[P5:[0-9]]]{{.*}}offset: 4{{.*}}size: 4
; CHECK: liveins: $r0, $r1, $r2, $r3
; CHECK: [[VREGR1:%[0-9]+]]:_(s32) = COPY $r1
; CHECK: [[VREGP1:%[0-9]+]]:_(s16) = G_TRUNC [[VREGR1]]
; CHECK: [[FIP5:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[P5]]
; CHECK: [[VREGP5EXT:%[0-9]+]]:_(s32) = G_LOAD [[FIP5]](p0){{.*}}load (s32)
; CHECK: [[ASSERT_SEXT:%[0-9]+]]:_(s32) = G_ASSERT_SEXT [[VREGP5EXT]], 16
; CHECK: [[VREGP5:%[0-9]+]]:_(s16) = G_TRUNC [[ASSERT_SEXT]]
; CHECK: [[SUM:%[0-9]+]]:_(s16) = G_ADD [[VREGP1]], [[VREGP5]]
; CHECK: [[SUM_EXT:%[0-9]+]]:_(s32) = G_ANYEXT [[SUM]]
; CHECK: $r0 = COPY [[SUM_EXT]](s32)
; CHECK: BX_RET 14 /* CC::al */, $noreg, implicit $r0
entry:
  %sum = add i16 %p1, %p5
  ret i16 %sum
}

define i8 @test_stack_args_zeroext(i32 %p0, i16 %p1, i8 %p2, i1 %p3,
                                    i8 zeroext %p4, i16 zeroext %p5) {
; CHECK-LABEL: name: test_stack_args_zeroext
; CHECK: fixedStack:
; CHECK-DAG: id: [[P4:[0-9]]]{{.*}}offset: 0{{.*}}size: 4
; CHECK-DAG: id: [[P5:[0-9]]]{{.*}}offset: 4{{.*}}size: 4
; CHECK: liveins: $r0, $r1, $r2, $r3
; CHECK: [[VREGR2:%[0-9]+]]:_(s32) = COPY $r2
; CHECK: [[VREGP2:%[0-9]+]]:_(s8) = G_TRUNC [[VREGR2]]
; CHECK: [[FIP4:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[P4]]
; CHECK: [[VREGP4EXT:%[0-9]+]]:_(s32) = G_LOAD [[FIP4]](p0){{.*}}load (s32)
; CHECK: [[ASSERT_ZEXT:%[0-9]+]]:_(s32) = G_ASSERT_ZEXT [[VREGP4EXT]], 8
; CHECK: [[VREGP4:%[0-9]+]]:_(s8) = G_TRUNC [[ASSERT_ZEXT]]
; CHECK: [[SUM:%[0-9]+]]:_(s8) = G_ADD [[VREGP2]], [[VREGP4]]
; CHECK: [[SUM_EXT:%[0-9]+]]:_(s32) = G_ANYEXT [[SUM]]
; CHECK: $r0 = COPY [[SUM_EXT]](s32)
; CHECK: BX_RET 14 /* CC::al */, $noreg, implicit $r0
entry:
  %sum = add i8 %p2, %p4
  ret i8 %sum
}

define i8 @test_stack_args_noext(i32 %p0, i16 %p1, i8 %p2, i1 %p3,
                                 i8 %p4, i16 %p5) {
; CHECK-LABEL: name: test_stack_args_noext
; CHECK: fixedStack:
; CHECK-DAG: id: [[P4:[0-9]]]{{.*}}offset: 0, size: 4, alignment: 8,
; CHECK-DAG: id: [[P5:[0-9]]]{{.*}}offset: 4, size: 4, alignment: 4,
; CHECK: liveins: $r0, $r1, $r2, $r3
; CHECK: [[VREGR2:%[0-9]+]]:_(s32) = COPY $r2
; CHECK: [[VREGP2:%[0-9]+]]:_(s8) = G_TRUNC [[VREGR2]]
; CHECK: [[FIP4:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[P4]]
; CHECK: [[VREGP4:%[0-9]+]]:_(s32) = G_LOAD [[FIP4]](p0){{.*}}load (s32)
; CHECK: [[TRUNC_VREGP4:%[0-9]+]]:_(s8) = G_TRUNC [[VREGP4]]
; CHECK: [[SUM:%[0-9]+]]:_(s8) = G_ADD [[VREGP2]], [[TRUNC_VREGP4]]
; CHECK: [[SUM_EXT:%[0-9]+]]:_(s32) = G_ANYEXT [[SUM]]
; CHECK: $r0 = COPY [[SUM_EXT]](s32)
; CHECK: BX_RET 14 /* CC::al */, $noreg, implicit $r0
entry:
  %sum = add i8 %p2, %p4
  ret i8 %sum
}

define zeroext i16 @test_stack_args_extend_the_extended(i32 %p0, i16 %p1, i8 %p2, i1 %p3,
                                                        i8 signext %p4, i16 signext %p5) {
; CHECK-LABEL: name: test_stack_args_extend_the_extended
; CHECK: fixedStack:
; CHECK-DAG: id: [[P4:[0-9]]]{{.*}}offset: 0{{.*}}size: 4, alignment: 8
; CHECK-DAG: id: [[P5:[0-9]]]{{.*}}offset: 4{{.*}}size: 4, alignment: 4
; CHECK: liveins: $r0, $r1, $r2, $r3
; CHECK: [[FIP5:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[P5]]
; CHECK: [[VREGP5SEXT:%[0-9]+]]:_(s32) = G_LOAD [[FIP5]](p0){{.*}}load (s32)
; CHECK: [[ASSERT_SEXT:%[0-9]+]]:_(s32) = G_ASSERT_SEXT [[VREGP5SEXT]], 16
; CHECK: [[VREGP5:%[0-9]+]]:_(s16) = G_TRUNC [[ASSERT_SEXT]]
; CHECK: [[VREGP5ZEXT:%[0-9]+]]:_(s32) = G_ZEXT [[VREGP5]]
; CHECK: $r0 = COPY [[VREGP5ZEXT]]
; CHECK: BX_RET 14 /* CC::al */, $noreg, implicit $r0
entry:
  ret i16 %p5
}

define i16 @test_ptr_arg(i16* %p) {
; CHECK-LABEL: name: test_ptr_arg
; CHECK: liveins: $r0
; CHECK: [[VREGP:%[0-9]+]]:_(p0) = COPY $r0
; CHECK: [[VREGV:%[0-9]+]]:_(s16) = G_LOAD [[VREGP]](p0){{.*}}load (s16)
entry:
  %v = load i16, i16* %p
  ret i16 %v
}

define i32* @test_ptr_ret(i32** %p) {
; Test pointer returns and pointer-to-pointer arguments
; CHECK-LABEL: name: test_ptr_ret
; CHECK: liveins: $r0
; CHECK: [[VREGP:%[0-9]+]]:_(p0) = COPY $r0
; CHECK: [[VREGV:%[0-9]+]]:_(p0) = G_LOAD [[VREGP]](p0){{.*}}load (p0)
; CHECK: $r0 = COPY [[VREGV]]
; CHECK: BX_RET 14 /* CC::al */, $noreg, implicit $r0
entry:
  %v = load i32*, i32** %p
  ret i32* %v
}

define i32 @test_ptr_arg_on_stack(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32* %p) {
; CHECK-LABEL: name: test_ptr_arg_on_stack
; CHECK: fixedStack:
; CHECK: id: [[P:[0-9]+]]{{.*}}offset: 0{{.*}}size: 4
; CHECK: liveins: $r0, $r1, $r2, $r3
; CHECK: [[FIP:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[P]]
; CHECK: [[VREGP:%[0-9]+]]:_(p0) = G_LOAD [[FIP]](p0){{.*}}load (p0)
; CHECK: [[VREGV:%[0-9]+]]:_(s32) = G_LOAD [[VREGP]](p0){{.*}}load (s32)
; CHECK: $r0 = COPY [[VREGV]]
; CHECK: BX_RET 14 /* CC::al */, $noreg, implicit $r0
entry:
  %v = load i32, i32* %p
  ret i32 %v
}

define arm_aapcscc float @test_float_aapcscc(float %p0, float %p1, float %p2,
                                             float %p3, float %p4, float %p5) {
; CHECK-LABEL: name: test_float_aapcscc
; CHECK: fixedStack:
; CHECK-DAG: id: [[P4:[0-9]+]]{{.*}}offset: 0{{.*}}size: 4
; CHECK-DAG: id: [[P5:[0-9]+]]{{.*}}offset: 4{{.*}}size: 4
; CHECK: liveins: $r0, $r1, $r2, $r3
; CHECK: [[VREGP1:%[0-9]+]]:_(s32) = COPY $r1
; CHECK: [[FIP5:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[P5]]
; CHECK: [[VREGP5:%[0-9]+]]:_(s32) = G_LOAD [[FIP5]](p0){{.*}}load (s32)
; CHECK: [[VREGV:%[0-9]+]]:_(s32) = G_FADD [[VREGP1]], [[VREGP5]]
; CHECK: $r0 = COPY [[VREGV]]
; CHECK: BX_RET 14 /* CC::al */, $noreg, implicit $r0
entry:
  %v = fadd float %p1, %p5
  ret float %v
}

define arm_aapcs_vfpcc float @test_float_vfpcc(float %p0, float %p1, float %p2,
                                               float %p3, float %p4, float %p5,
                                               float %ridiculous,
                                               float %number,
                                               float %of,
                                               float %parameters,
                                               float %that,
                                               float %should,
                                               float %never,
                                               float %exist,
                                               float %in,
                                               float %practice,
                                               float %q0, float %q1) {
; CHECK-LABEL: name: test_float_vfpcc
; CHECK: fixedStack:
; CHECK-DAG: id: [[Q0:[0-9]+]]{{.*}}offset: 0{{.*}}size: 4
; CHECK-DAG: id: [[Q1:[0-9]+]]{{.*}}offset: 4{{.*}}size: 4
; CHECK: liveins: $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7, $s8, $s9, $s10, $s11, $s12, $s13, $s14, $s15
; CHECK: [[VREGP1:%[0-9]+]]:_(s32) = COPY $s1
; CHECK: [[FIQ1:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[Q1]]
; CHECK: [[VREGQ1:%[0-9]+]]:_(s32) = G_LOAD [[FIQ1]](p0){{.*}}load (s32)
; CHECK: [[VREGV:%[0-9]+]]:_(s32) = G_FADD [[VREGP1]], [[VREGQ1]]
; CHECK: $s0 = COPY [[VREGV]]
; CHECK: BX_RET 14 /* CC::al */, $noreg, implicit $s0
entry:
  %v = fadd float %p1, %q1
  ret float %v
}

define arm_aapcs_vfpcc double @test_double_vfpcc(double %p0, double %p1, double %p2,
                                                 double %p3, double %p4, double %p5,
                                                 double %reasonable,
                                                 double %parameters,
                                                 double %q0, double %q1) {
; CHECK-LABEL: name: test_double_vfpcc
; CHECK: fixedStack:
; CHECK-DAG: id: [[Q0:[0-9]+]]{{.*}}offset: 0{{.*}}size: 8
; CHECK-DAG: id: [[Q1:[0-9]+]]{{.*}}offset: 8{{.*}}size: 8
; CHECK: liveins: $d0, $d1, $d2, $d3, $d4, $d5, $d6, $d7
; CHECK: [[VREGP1:%[0-9]+]]:_(s64) = COPY $d1
; CHECK: [[FIQ1:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[Q1]]
; CHECK: [[VREGQ1:%[0-9]+]]:_(s64) = G_LOAD [[FIQ1]](p0){{.*}}load (s64)
; CHECK: [[VREGV:%[0-9]+]]:_(s64) = G_FADD [[VREGP1]], [[VREGQ1]]
; CHECK: $d0 = COPY [[VREGV]]
; CHECK: BX_RET 14 /* CC::al */, $noreg, implicit $d0
entry:
  %v = fadd double %p1, %q1
  ret double %v
}

define arm_aapcscc double @test_double_aapcscc(double %p0, double %p1, double %p2,
                                               double %p3, double %p4, double %p5) {
; CHECK-LABEL: name: test_double_aapcscc
; CHECK: fixedStack:
; CHECK-DAG: id: [[P2:[0-9]+]]{{.*}}offset: 0{{.*}}size: 8
; CHECK-DAG: id: [[P3:[0-9]+]]{{.*}}offset: 8{{.*}}size: 8
; CHECK-DAG: id: [[P4:[0-9]+]]{{.*}}offset: 16{{.*}}size: 8
; CHECK-DAG: id: [[P5:[0-9]+]]{{.*}}offset: 24{{.*}}size: 8
; CHECK: liveins: $r0, $r1, $r2, $r3
; CHECK-DAG: [[VREGP1LO:%[0-9]+]]:_(s32) = COPY $r2
; CHECK-DAG: [[VREGP1HI:%[0-9]+]]:_(s32) = COPY $r3
; LITTLE: [[VREGP1:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[VREGP1LO]](s32), [[VREGP1HI]](s32)
; BIG: [[VREGP1:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[VREGP1HI]](s32), [[VREGP1LO]](s32)
; CHECK: [[FIP5:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[P5]]
; CHECK: [[VREGP5:%[0-9]+]]:_(s64) = G_LOAD [[FIP5]](p0){{.*}}load (s64)
; CHECK: [[VREGV:%[0-9]+]]:_(s64) = G_FADD [[VREGP1]], [[VREGP5]]
; LITTLE: [[VREGVLO:%[0-9]+]]:_(s32), [[VREGVHI:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[VREGV]](s64)
; BIG: [[VREGVHI:%[0-9]+]]:_(s32), [[VREGVLO:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[VREGV]](s64)
; CHECK-DAG: $r0 = COPY [[VREGVLO]]
; CHECK-DAG: $r1 = COPY [[VREGVHI]]
; CHECK: BX_RET 14 /* CC::al */, $noreg, implicit $r0, implicit $r1
entry:
  %v = fadd double %p1, %p5
  ret double %v
}

define arm_aapcs_vfpcc double @test_double_gap_vfpcc(double %p0, float %filler,
                                                     double %p1, double %p2,
                                                     double %p3, double %p4,
                                                     double %reasonable,
                                                     double %parameters,
                                                     double %q0, double %q1) {
; CHECK-LABEL: name: test_double_gap_vfpcc
; CHECK: fixedStack:
; CHECK-DAG: id: [[Q0:[0-9]+]]{{.*}}offset: 0{{.*}}size: 8
; CHECK-DAG: id: [[Q1:[0-9]+]]{{.*}}offset: 8{{.*}}size: 8
; CHECK: liveins: $d0, $d2, $d3, $d4, $d5, $d6, $d7, $s2
; CHECK: [[VREGP1:%[0-9]+]]:_(s64) = COPY $d2
; CHECK: [[FIQ1:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[Q1]]
; CHECK: [[VREGQ1:%[0-9]+]]:_(s64) = G_LOAD [[FIQ1]](p0){{.*}}load (s64)
; CHECK: [[VREGV:%[0-9]+]]:_(s64) = G_FADD [[VREGP1]], [[VREGQ1]]
; CHECK: $d0 = COPY [[VREGV]]
; CHECK: BX_RET 14 /* CC::al */, $noreg, implicit $d0
entry:
  %v = fadd double %p1, %q1
  ret double %v
}

define arm_aapcscc double @test_double_gap_aapcscc(float %filler, double %p0,
                                                   double %p1) {
; CHECK-LABEL: name: test_double_gap_aapcscc
; CHECK: fixedStack:
; CHECK-DAG: id: [[P1:[0-9]+]]{{.*}}offset: 0{{.*}}size: 8
; CHECK: liveins: $r0, $r2, $r3
; CHECK-DAG: [[VREGP0LO:%[0-9]+]]:_(s32) = COPY $r2
; CHECK-DAG: [[VREGP0HI:%[0-9]+]]:_(s32) = COPY $r3
; LITTLE: [[VREGP0:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[VREGP0LO]](s32), [[VREGP0HI]](s32)
; BIG: [[VREGP0:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[VREGP0HI]](s32), [[VREGP0LO]](s32)
; CHECK: [[FIP1:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[P1]]
; CHECK: [[VREGP1:%[0-9]+]]:_(s64) = G_LOAD [[FIP1]](p0){{.*}}load (s64)
; CHECK: [[VREGV:%[0-9]+]]:_(s64) = G_FADD [[VREGP0]], [[VREGP1]]
; LITTLE: [[VREGVLO:%[0-9]+]]:_(s32), [[VREGVHI:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[VREGV]](s64)
; BIG: [[VREGVHI:%[0-9]+]]:_(s32), [[VREGVLO:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[VREGV]](s64)
; CHECK-DAG: $r0 = COPY [[VREGVLO]]
; CHECK-DAG: $r1 = COPY [[VREGVHI]]
; CHECK: BX_RET 14 /* CC::al */, $noreg, implicit $r0, implicit $r1
entry:
  %v = fadd double %p0, %p1
  ret double %v
}

define arm_aapcscc double @test_double_gap2_aapcscc(double %p0, float %filler,
                                                    double %p1) {
; CHECK-LABEL: name: test_double_gap2_aapcscc
; CHECK: fixedStack:
; CHECK-DAG: id: [[P1:[0-9]+]]{{.*}}offset: 0{{.*}}size: 8
; CHECK: liveins: $r0, $r1, $r2
; CHECK-DAG: [[VREGP0LO:%[0-9]+]]:_(s32) = COPY $r0
; CHECK-DAG: [[VREGP0HI:%[0-9]+]]:_(s32) = COPY $r1
; LITTLE: [[VREGP0:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[VREGP0LO]](s32), [[VREGP0HI]](s32)
; BIG: [[VREGP0:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[VREGP0HI]](s32), [[VREGP0LO]](s32)
; CHECK: [[FIP1:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[P1]]
; CHECK: [[VREGP1:%[0-9]+]]:_(s64) = G_LOAD [[FIP1]](p0){{.*}}load (s64)
; CHECK: [[VREGV:%[0-9]+]]:_(s64) = G_FADD [[VREGP0]], [[VREGP1]]
; LITTLE: [[VREGVLO:%[0-9]+]]:_(s32), [[VREGVHI:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[VREGV]](s64)
; BIG: [[VREGVHI:%[0-9]+]]:_(s32), [[VREGVLO:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[VREGV]](s64)
; CHECK-DAG: $r0 = COPY [[VREGVLO]]
; CHECK-DAG: $r1 = COPY [[VREGVHI]]
; CHECK: BX_RET 14 /* CC::al */, $noreg, implicit $r0, implicit $r1
entry:
  %v = fadd double %p0, %p1
  ret double %v
}

define i32 @test_shufflevector_s32_v2s32(i32 %arg) {
; CHECK-LABEL: name: test_shufflevector_s32_v2s32
; CHECK: [[ARG:%[0-9]+]]:_(s32) = COPY $r0
; CHECK-DAG: [[UNDEF:%[0-9]+]]:_(s32) = G_IMPLICIT_DEF
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s32>) = G_SHUFFLE_VECTOR [[ARG]](s32), [[UNDEF]], shufflemask(0, 0)
; CHECK: G_EXTRACT_VECTOR_ELT [[VEC]](<2 x s32>)
  %vec = insertelement <1 x i32> undef, i32 %arg, i32 0
  %shuffle = shufflevector <1 x i32> %vec, <1 x i32> undef, <2 x i32> zeroinitializer
  %res = extractelement <2 x i32> %shuffle, i32 0
  ret i32 %res
}

define i32 @test_shufflevector_s32_s32_s32(i32 %arg) {
; CHECK-LABEL: name: test_shufflevector_s32_s32_s32
; CHECK: [[ARG:%[0-9]+]]:_(s32) = COPY $r0
; CHECK-DAG: [[UNDEF:%[0-9]+]]:_(s32) = G_IMPLICIT_DEF
; CHECK: [[VEC:%[0-9]+]]:_(s32) = G_SHUFFLE_VECTOR [[ARG]](s32), [[UNDEF]], shufflemask(0)
  %vec = insertelement <1 x i32> undef, i32 %arg, i32 0
  %shuffle = shufflevector <1 x i32> %vec, <1 x i32> undef, <1 x i32> zeroinitializer
  %res = extractelement <1 x i32> %shuffle, i32 0
  ret i32 %res
}

define i32 @test_shufflevector_v2s32_v3s32(i32 %arg1, i32 %arg2) {
; CHECK-LABEL: name: test_shufflevector_v2s32_v3s32
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY $r0
; CHECK: [[ARG2:%[0-9]+]]:_(s32) = COPY $r1
; CHECK-DAG: [[UNDEF:%[0-9]+]]:_(<2 x s32>) = G_IMPLICIT_DEF
; CHECK-DAG: [[C0:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK-DAG: [[C1:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK-DAG: [[V1:%[0-9]+]]:_(<2 x s32>) = G_INSERT_VECTOR_ELT [[UNDEF]], [[ARG1]](s32), [[C0]](s32)
; CHECK-DAG: [[V2:%[0-9]+]]:_(<2 x s32>) = G_INSERT_VECTOR_ELT [[V1]], [[ARG2]](s32), [[C1]](s32)
; CHECK: [[VEC:%[0-9]+]]:_(<3 x s32>) = G_SHUFFLE_VECTOR [[V2]](<2 x s32>), [[UNDEF]], shufflemask(1, 0, 1)
; CHECK: G_EXTRACT_VECTOR_ELT [[VEC]](<3 x s32>)
  %v1 = insertelement <2 x i32> undef, i32 %arg1, i32 0
  %v2 = insertelement <2 x i32> %v1, i32 %arg2, i32 1
  %shuffle = shufflevector <2 x i32> %v2, <2 x i32> undef, <3 x i32> <i32 1, i32 0, i32 1>
  %res = extractelement <3 x i32> %shuffle, i32 0
  ret i32 %res
}


define i32 @test_shufflevector_v2s32_v4s32(i32 %arg1, i32 %arg2) {
; CHECK-LABEL: name: test_shufflevector_v2s32_v4s32
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY $r0
; CHECK: [[ARG2:%[0-9]+]]:_(s32) = COPY $r1
; CHECK-DAG: [[UNDEF:%[0-9]+]]:_(<2 x s32>) = G_IMPLICIT_DEF
; CHECK-DAG: [[C0:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK-DAG: [[C1:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK-DAG: [[V1:%[0-9]+]]:_(<2 x s32>) = G_INSERT_VECTOR_ELT [[UNDEF]], [[ARG1]](s32), [[C0]](s32)
; CHECK-DAG: [[V2:%[0-9]+]]:_(<2 x s32>) = G_INSERT_VECTOR_ELT [[V1]], [[ARG2]](s32), [[C1]](s32)
; CHECK: [[VEC:%[0-9]+]]:_(<4 x s32>) = G_SHUFFLE_VECTOR [[V2]](<2 x s32>), [[UNDEF]], shufflemask(0, 0, 0, 0)
; CHECK: G_EXTRACT_VECTOR_ELT [[VEC]](<4 x s32>)
  %v1 = insertelement <2 x i32> undef, i32 %arg1, i32 0
  %v2 = insertelement <2 x i32> %v1, i32 %arg2, i32 1
  %shuffle = shufflevector <2 x i32> %v2, <2 x i32> undef, <4 x i32> zeroinitializer
  %res = extractelement <4 x i32> %shuffle, i32 0
  ret i32 %res
}

define i32 @test_shufflevector_v4s32_v2s32(i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4) {
; CHECK-LABEL: name: test_shufflevector_v4s32_v2s32
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY $r0
; CHECK: [[ARG2:%[0-9]+]]:_(s32) = COPY $r1
; CHECK: [[ARG3:%[0-9]+]]:_(s32) = COPY $r2
; CHECK: [[ARG4:%[0-9]+]]:_(s32) = COPY $r3
; CHECK-DAG: [[UNDEF:%[0-9]+]]:_(<4 x s32>) = G_IMPLICIT_DEF
; CHECK-DAG: [[C0:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK-DAG: [[C1:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK-DAG: [[C2:%[0-9]+]]:_(s32) = G_CONSTANT i32 2
; CHECK-DAG: [[C3:%[0-9]+]]:_(s32) = G_CONSTANT i32 3
; CHECK-DAG: [[V1:%[0-9]+]]:_(<4 x s32>) = G_INSERT_VECTOR_ELT [[UNDEF]], [[ARG1]](s32), [[C0]](s32)
; CHECK-DAG: [[V2:%[0-9]+]]:_(<4 x s32>) = G_INSERT_VECTOR_ELT [[V1]], [[ARG2]](s32), [[C1]](s32)
; CHECK-DAG: [[V3:%[0-9]+]]:_(<4 x s32>) = G_INSERT_VECTOR_ELT [[V2]], [[ARG3]](s32), [[C2]](s32)
; CHECK-DAG: [[V4:%[0-9]+]]:_(<4 x s32>) = G_INSERT_VECTOR_ELT [[V3]], [[ARG4]](s32), [[C3]](s32)
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s32>) = G_SHUFFLE_VECTOR [[V4]](<4 x s32>), [[UNDEF]], shufflemask(1, 3)
; CHECK: G_EXTRACT_VECTOR_ELT [[VEC]](<2 x s32>)
  %v1 = insertelement <4 x i32> undef, i32 %arg1, i32 0
  %v2 = insertelement <4 x i32> %v1, i32 %arg2, i32 1
  %v3 = insertelement <4 x i32> %v2, i32 %arg3, i32 2
  %v4 = insertelement <4 x i32> %v3, i32 %arg4, i32 3
  %shuffle = shufflevector <4 x i32> %v4, <4 x i32> undef, <2 x i32> <i32 1, i32 3>
  %res = extractelement <2 x i32> %shuffle, i32 0
  ret i32 %res
}

%struct.v2s32 = type { <2 x i32> }

define i32 @test_constantstruct_v2s32() {
; CHECK-LABEL: name: test_constantstruct_v2s32
; CHECK: [[C1:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK: [[C2:%[0-9]+]]:_(s32) = G_CONSTANT i32 2
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s32>) = G_BUILD_VECTOR [[C1]](s32), [[C2]](s32)
; CHECK: [[C3:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK: G_EXTRACT_VECTOR_ELT [[VEC]](<2 x s32>), [[C3]]
  %vec = extractvalue %struct.v2s32 {<2 x i32><i32 1, i32 2>}, 0
  %elt = extractelement <2 x i32> %vec, i32 0
  ret i32 %elt
}

%struct.v2s32.s32.s32 = type { <2 x i32>, i32, i32 }

define i32 @test_constantstruct_v2s32_s32_s32() {
; CHECK-LABEL: name: test_constantstruct_v2s32_s32_s32
; CHECK: [[C1:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK: [[C2:%[0-9]+]]:_(s32) = G_CONSTANT i32 2
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s32>) = G_BUILD_VECTOR [[C1]](s32), [[C2]](s32)
; CHECK: [[C3:%[0-9]+]]:_(s32) = G_CONSTANT i32 3
; CHECK: [[C4:%[0-9]+]]:_(s32) = G_CONSTANT i32 4
; CHECK: [[C5:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK: G_EXTRACT_VECTOR_ELT [[VEC]](<2 x s32>), [[C5]](s32)
  %vec = extractvalue %struct.v2s32.s32.s32 {<2 x i32><i32 1, i32 2>, i32 3, i32 4}, 0
  %elt = extractelement <2 x i32> %vec, i32 0
  ret i32 %elt
}

define void @test_load_store_struct({i32, i32} *%addr) {
; Make sure the IRTranslator doesn't use an unnecessarily large GEP index type
; when breaking up loads and stores of aggregates.
; CHECK-LABEL: name: test_load_store_struct
; CHECK: [[ADDR1:%[0-9]+]]:_(p0) = COPY $r0
; CHECK-DAG: [[VAL1:%[0-9]+]]:_(s32) = G_LOAD [[ADDR1]](p0) :: (load (s32) from %ir.addr)
; CHECK-DAG: [[OFFSET:%[0-9]+]]:_(s32) = G_CONSTANT i32 4
; CHECK-DAG: [[ADDR2:%[0-9]+]]:_(p0) = G_PTR_ADD [[ADDR1]], [[OFFSET]](s32)
; CHECK-DAG: [[VAL2:%[0-9]+]]:_(s32) = G_LOAD [[ADDR2]](p0) :: (load (s32) from %ir.addr + 4)
; CHECK-DAG: G_STORE [[VAL1]](s32), [[ADDR1]](p0) :: (store (s32) into %ir.addr)
; CHECK-DAG: [[ADDR3:%[0-9]+]]:_(p0) = COPY [[ADDR2]]
; CHECK-DAG: G_STORE [[VAL2]](s32), [[ADDR3]](p0) :: (store (s32) into %ir.addr + 4)
  %val = load {i32, i32}, {i32, i32} *%addr, align 4
  store {i32, i32} %val, {i32, i32} *%addr, align 4
  ret void
}
