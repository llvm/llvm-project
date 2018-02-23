; RUN: llc -mtriple arm-unknown -mattr=+vfp2 -global-isel -stop-after=irtranslator -verify-machineinstrs %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=LITTLE
; RUN: llc -mtriple armeb-unknown -mattr=+vfp2 -global-isel -stop-after=irtranslator -verify-machineinstrs %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=BIG

define void @test_void_return() {
; CHECK-LABEL: name: test_void_return
; CHECK: BX_RET 14, _
entry:
  ret void
}

define signext i1 @test_add_i1(i1 %x, i1 %y) {
; CHECK-LABEL: name: test_add_i1
; CHECK: liveins: %r0, %r1
; CHECK-DAG: [[VREGX:%[0-9]+]](s1) = COPY %r0
; CHECK-DAG: [[VREGY:%[0-9]+]](s1) = COPY %r1
; CHECK: [[SUM:%[0-9]+]](s1) = G_ADD [[VREGX]], [[VREGY]]
; CHECK: [[EXT:%[0-9]+]](s32) = G_SEXT [[SUM]]
; CHECK: %r0 = COPY [[EXT]](s32)
; CHECK: BX_RET 14, _, implicit %r0
entry:
  %sum = add i1 %x, %y
  ret i1 %sum
}

define i8 @test_add_i8(i8 %x, i8 %y) {
; CHECK-LABEL: name: test_add_i8
; CHECK: liveins: %r0, %r1
; CHECK-DAG: [[VREGX:%[0-9]+]](s8) = COPY %r0
; CHECK-DAG: [[VREGY:%[0-9]+]](s8) = COPY %r1
; CHECK: [[SUM:%[0-9]+]](s8) = G_ADD [[VREGX]], [[VREGY]]
; CHECK: %r0 = COPY [[SUM]](s8)
; CHECK: BX_RET 14, _, implicit %r0
entry:
  %sum = add i8 %x, %y
  ret i8 %sum
}

define i8 @test_sub_i8(i8 %x, i8 %y) {
; CHECK-LABEL: name: test_sub_i8
; CHECK: liveins: %r0, %r1
; CHECK-DAG: [[VREGX:%[0-9]+]](s8) = COPY %r0
; CHECK-DAG: [[VREGY:%[0-9]+]](s8) = COPY %r1
; CHECK: [[RES:%[0-9]+]](s8) = G_SUB [[VREGX]], [[VREGY]]
; CHECK: %r0 = COPY [[RES]](s8)
; CHECK: BX_RET 14, _, implicit %r0
entry:
  %res = sub i8 %x, %y
  ret i8 %res
}

define signext i8 @test_return_sext_i8(i8 %x) {
; CHECK-LABEL: name: test_return_sext_i8
; CHECK: liveins: %r0
; CHECK: [[VREG:%[0-9]+]](s8) = COPY %r0
; CHECK: [[VREGEXT:%[0-9]+]](s32) = G_SEXT [[VREG]]
; CHECK: %r0 = COPY [[VREGEXT]](s32)
; CHECK: BX_RET 14, _, implicit %r0
entry:
  ret i8 %x
}

define i16 @test_add_i16(i16 %x, i16 %y) {
; CHECK-LABEL: name: test_add_i16
; CHECK: liveins: %r0, %r1
; CHECK-DAG: [[VREGX:%[0-9]+]](s16) = COPY %r0
; CHECK-DAG: [[VREGY:%[0-9]+]](s16) = COPY %r1
; CHECK: [[SUM:%[0-9]+]](s16) = G_ADD [[VREGX]], [[VREGY]]
; CHECK: %r0 = COPY [[SUM]](s16)
; CHECK: BX_RET 14, _, implicit %r0
entry:
  %sum = add i16 %x, %y
  ret i16 %sum
}

define i16 @test_sub_i16(i16 %x, i16 %y) {
; CHECK-LABEL: name: test_sub_i16
; CHECK: liveins: %r0, %r1
; CHECK-DAG: [[VREGX:%[0-9]+]](s16) = COPY %r0
; CHECK-DAG: [[VREGY:%[0-9]+]](s16) = COPY %r1
; CHECK: [[RES:%[0-9]+]](s16) = G_SUB [[VREGX]], [[VREGY]]
; CHECK: %r0 = COPY [[RES]](s16)
; CHECK: BX_RET 14, _, implicit %r0
entry:
  %res = sub i16 %x, %y
  ret i16 %res
}

define zeroext i16 @test_return_zext_i16(i16 %x) {
; CHECK-LABEL: name: test_return_zext_i16
; CHECK: liveins: %r0
; CHECK: [[VREG:%[0-9]+]](s16) = COPY %r0
; CHECK: [[VREGEXT:%[0-9]+]](s32) = G_ZEXT [[VREG]]
; CHECK: %r0 = COPY [[VREGEXT]](s32)
; CHECK: BX_RET 14, _, implicit %r0
entry:
  ret i16 %x
}

define i32 @test_add_i32(i32 %x, i32 %y) {
; CHECK-LABEL: name: test_add_i32
; CHECK: liveins: %r0, %r1
; CHECK-DAG: [[VREGX:%[0-9]+]](s32) = COPY %r0
; CHECK-DAG: [[VREGY:%[0-9]+]](s32) = COPY %r1
; CHECK: [[SUM:%[0-9]+]](s32) = G_ADD [[VREGX]], [[VREGY]]
; CHECK: %r0 = COPY [[SUM]](s32)
; CHECK: BX_RET 14, _, implicit %r0
entry:
  %sum = add i32 %x, %y
  ret i32 %sum
}

define i32 @test_sub_i32(i32 %x, i32 %y) {
; CHECK-LABEL: name: test_sub_i32
; CHECK: liveins: %r0, %r1
; CHECK-DAG: [[VREGX:%[0-9]+]](s32) = COPY %r0
; CHECK-DAG: [[VREGY:%[0-9]+]](s32) = COPY %r1
; CHECK: [[RES:%[0-9]+]](s32) = G_SUB [[VREGX]], [[VREGY]]
; CHECK: %r0 = COPY [[RES]](s32)
; CHECK: BX_RET 14, _, implicit %r0
entry:
  %res = sub i32 %x, %y
  ret i32 %res
}

define i32 @test_stack_args(i32 %p0, i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5) {
; CHECK-LABEL: name: test_stack_args
; CHECK: fixedStack:
; CHECK-DAG: id: [[P4:[0-9]]]{{.*}}offset: 0{{.*}}size: 4
; CHECK-DAG: id: [[P5:[0-9]]]{{.*}}offset: 4{{.*}}size: 4
; CHECK: liveins: %r0, %r1, %r2, %r3
; CHECK: [[VREGP2:%[0-9]+]](s32) = COPY %r2
; CHECK: [[FIP5:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[P5]]
; CHECK: [[VREGP5:%[0-9]+]](s32) = G_LOAD [[FIP5]]{{.*}}load 4
; CHECK: [[SUM:%[0-9]+]](s32) = G_ADD [[VREGP2]], [[VREGP5]]
; CHECK: %r0 = COPY [[SUM]]
; CHECK: BX_RET 14, _, implicit %r0
entry:
  %sum = add i32 %p2, %p5
  ret i32 %sum
}

define i16 @test_stack_args_signext(i32 %p0, i16 %p1, i8 %p2, i1 %p3,
                                    i8 signext %p4, i16 signext %p5) {
; CHECK-LABEL: name: test_stack_args_signext
; CHECK: fixedStack:
; CHECK-DAG: id: [[P4:[0-9]]]{{.*}}offset: 0{{.*}}size: 1
; CHECK-DAG: id: [[P5:[0-9]]]{{.*}}offset: 4{{.*}}size: 2
; CHECK: liveins: %r0, %r1, %r2, %r3
; CHECK: [[VREGP1:%[0-9]+]](s16) = COPY %r1
; CHECK: [[FIP5:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[P5]]
; CHECK: [[VREGP5EXT:%[0-9]+]](s32) = G_LOAD [[FIP5]](p0){{.*}}load 4
; CHECK: [[VREGP5:%[0-9]+]](s16) = G_TRUNC [[VREGP5EXT]]
; CHECK: [[SUM:%[0-9]+]](s16) = G_ADD [[VREGP1]], [[VREGP5]]
; CHECK: %r0 = COPY [[SUM]]
; CHECK: BX_RET 14, _, implicit %r0
entry:
  %sum = add i16 %p1, %p5
  ret i16 %sum
}

define i8 @test_stack_args_zeroext(i32 %p0, i16 %p1, i8 %p2, i1 %p3,
                                    i8 zeroext %p4, i16 zeroext %p5) {
; CHECK-LABEL: name: test_stack_args_zeroext
; CHECK: fixedStack:
; CHECK-DAG: id: [[P4:[0-9]]]{{.*}}offset: 0{{.*}}size: 1
; CHECK-DAG: id: [[P5:[0-9]]]{{.*}}offset: 4{{.*}}size: 2
; CHECK: liveins: %r0, %r1, %r2, %r3
; CHECK: [[VREGP2:%[0-9]+]](s8) = COPY %r2
; CHECK: [[FIP4:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[P4]]
; CHECK: [[VREGP4EXT:%[0-9]+]](s32) = G_LOAD [[FIP4]](p0){{.*}}load 4
; CHECK: [[VREGP4:%[0-9]+]](s8) = G_TRUNC [[VREGP4EXT]]
; CHECK: [[SUM:%[0-9]+]](s8) = G_ADD [[VREGP2]], [[VREGP4]]
; CHECK: %r0 = COPY [[SUM]]
; CHECK: BX_RET 14, _, implicit %r0
entry:
  %sum = add i8 %p2, %p4
  ret i8 %sum
}

define i8 @test_stack_args_noext(i32 %p0, i16 %p1, i8 %p2, i1 %p3,
                                 i8 %p4, i16 %p5) {
; CHECK-LABEL: name: test_stack_args_noext
; CHECK: fixedStack:
; CHECK-DAG: id: [[P4:[0-9]]]{{.*}}offset: 0{{.*}}size: 1
; CHECK-DAG: id: [[P5:[0-9]]]{{.*}}offset: 4{{.*}}size: 2
; CHECK: liveins: %r0, %r1, %r2, %r3
; CHECK: [[VREGP2:%[0-9]+]](s8) = COPY %r2
; CHECK: [[FIP4:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[P4]]
; CHECK: [[VREGP4:%[0-9]+]](s8) = G_LOAD [[FIP4]](p0){{.*}}load 1
; CHECK: [[SUM:%[0-9]+]](s8) = G_ADD [[VREGP2]], [[VREGP4]]
; CHECK: %r0 = COPY [[SUM]]
; CHECK: BX_RET 14, _, implicit %r0
entry:
  %sum = add i8 %p2, %p4
  ret i8 %sum
}

define zeroext i16 @test_stack_args_extend_the_extended(i32 %p0, i16 %p1, i8 %p2, i1 %p3,
                                                        i8 signext %p4, i16 signext %p5) {
; CHECK-LABEL: name: test_stack_args_extend_the_extended
; CHECK: fixedStack:
; CHECK-DAG: id: [[P4:[0-9]]]{{.*}}offset: 0{{.*}}size: 1
; CHECK-DAG: id: [[P5:[0-9]]]{{.*}}offset: 4{{.*}}size: 2
; CHECK: liveins: %r0, %r1, %r2, %r3
; CHECK: [[FIP5:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[P5]]
; CHECK: [[VREGP5SEXT:%[0-9]+]](s32) = G_LOAD [[FIP5]](p0){{.*}}load 4
; CHECK: [[VREGP5:%[0-9]+]](s16) = G_TRUNC [[VREGP5SEXT]]
; CHECK: [[VREGP5ZEXT:%[0-9]+]](s32) = G_ZEXT [[VREGP5]]
; CHECK: %r0 = COPY [[VREGP5ZEXT]]
; CHECK: BX_RET 14, _, implicit %r0
entry:
  ret i16 %p5
}

define i16 @test_ptr_arg(i16* %p) {
; CHECK-LABEL: name: test_ptr_arg
; CHECK: liveins: %r0
; CHECK: [[VREGP:%[0-9]+]](p0) = COPY %r0
; CHECK: [[VREGV:%[0-9]+]](s16) = G_LOAD [[VREGP]](p0){{.*}}load 2
entry:
  %v = load i16, i16* %p
  ret i16 %v
}

define i32* @test_ptr_ret(i32** %p) {
; Test pointer returns and pointer-to-pointer arguments
; CHECK-LABEL: name: test_ptr_ret
; CHECK: liveins: %r0
; CHECK: [[VREGP:%[0-9]+]](p0) = COPY %r0
; CHECK: [[VREGV:%[0-9]+]](p0) = G_LOAD [[VREGP]](p0){{.*}}load 4
; CHECK: %r0 = COPY [[VREGV]]
; CHECK: BX_RET 14, _, implicit %r0
entry:
  %v = load i32*, i32** %p
  ret i32* %v
}

define i32 @test_ptr_arg_on_stack(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32* %p) {
; CHECK-LABEL: name: test_ptr_arg_on_stack
; CHECK: fixedStack:
; CHECK: id: [[P:[0-9]+]]{{.*}}offset: 0{{.*}}size: 4
; CHECK: liveins: %r0, %r1, %r2, %r3
; CHECK: [[FIP:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[P]]
; CHECK: [[VREGP:%[0-9]+]](p0) = G_LOAD [[FIP]](p0){{.*}}load 4
; CHECK: [[VREGV:%[0-9]+]](s32) = G_LOAD [[VREGP]](p0){{.*}}load 4
; CHECK: %r0 = COPY [[VREGV]]
; CHECK: BX_RET 14, _, implicit %r0
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
; CHECK: liveins: %r0, %r1, %r2, %r3
; CHECK: [[VREGP1:%[0-9]+]](s32) = COPY %r1
; CHECK: [[FIP5:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[P5]]
; CHECK: [[VREGP5:%[0-9]+]](s32) = G_LOAD [[FIP5]](p0){{.*}}load 4
; CHECK: [[VREGV:%[0-9]+]](s32) = G_FADD [[VREGP1]], [[VREGP5]]
; CHECK: %r0 = COPY [[VREGV]]
; CHECK: BX_RET 14, _, implicit %r0
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
; CHECK: liveins: %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %s9, %s10, %s11, %s12, %s13, %s14, %s15
; CHECK: [[VREGP1:%[0-9]+]](s32) = COPY %s1
; CHECK: [[FIQ1:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[Q1]]
; CHECK: [[VREGQ1:%[0-9]+]](s32) = G_LOAD [[FIQ1]](p0){{.*}}load 4
; CHECK: [[VREGV:%[0-9]+]](s32) = G_FADD [[VREGP1]], [[VREGQ1]]
; CHECK: %s0 = COPY [[VREGV]]
; CHECK: BX_RET 14, _, implicit %s0
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
; CHECK: liveins: %d0, %d1, %d2, %d3, %d4, %d5, %d6, %d7
; CHECK: [[VREGP1:%[0-9]+]](s64) = COPY %d1
; CHECK: [[FIQ1:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[Q1]]
; CHECK: [[VREGQ1:%[0-9]+]](s64) = G_LOAD [[FIQ1]](p0){{.*}}load 8
; CHECK: [[VREGV:%[0-9]+]](s64) = G_FADD [[VREGP1]], [[VREGQ1]]
; CHECK: %d0 = COPY [[VREGV]]
; CHECK: BX_RET 14, _, implicit %d0
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
; CHECK: liveins: %r0, %r1, %r2, %r3
; CHECK-DAG: [[VREGP1LO:%[0-9]+]](s32) = COPY %r2
; CHECK-DAG: [[VREGP1HI:%[0-9]+]](s32) = COPY %r3
; LITTLE: [[VREGP1:%[0-9]+]](s64) = G_MERGE_VALUES [[VREGP1LO]](s32), [[VREGP1HI]](s32)
; BIG: [[VREGP1:%[0-9]+]](s64) = G_MERGE_VALUES [[VREGP1HI]](s32), [[VREGP1LO]](s32)
; CHECK: [[FIP5:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[P5]]
; CHECK: [[VREGP5:%[0-9]+]](s64) = G_LOAD [[FIP5]](p0){{.*}}load 8
; CHECK: [[VREGV:%[0-9]+]](s64) = G_FADD [[VREGP1]], [[VREGP5]]
; LITTLE: [[VREGVLO:%[0-9]+]](s32), [[VREGVHI:%[0-9]+]](s32) = G_UNMERGE_VALUES [[VREGV]](s64)
; BIG: [[VREGVHI:%[0-9]+]](s32), [[VREGVLO:%[0-9]+]](s32) = G_UNMERGE_VALUES [[VREGV]](s64)
; CHECK-DAG: %r0 = COPY [[VREGVLO]]
; CHECK-DAG: %r1 = COPY [[VREGVHI]]
; CHECK: BX_RET 14, _, implicit %r0, implicit %r1
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
; CHECK: liveins: %d0, %d2, %d3, %d4, %d5, %d6, %d7, %s2
; CHECK: [[VREGP1:%[0-9]+]](s64) = COPY %d2
; CHECK: [[FIQ1:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[Q1]]
; CHECK: [[VREGQ1:%[0-9]+]](s64) = G_LOAD [[FIQ1]](p0){{.*}}load 8
; CHECK: [[VREGV:%[0-9]+]](s64) = G_FADD [[VREGP1]], [[VREGQ1]]
; CHECK: %d0 = COPY [[VREGV]]
; CHECK: BX_RET 14, _, implicit %d0
entry:
  %v = fadd double %p1, %q1
  ret double %v
}

define arm_aapcscc double @test_double_gap_aapcscc(float %filler, double %p0,
                                                   double %p1) {
; CHECK-LABEL: name: test_double_gap_aapcscc
; CHECK: fixedStack:
; CHECK-DAG: id: [[P1:[0-9]+]]{{.*}}offset: 0{{.*}}size: 8
; CHECK: liveins: %r0, %r2, %r3
; CHECK-DAG: [[VREGP0LO:%[0-9]+]](s32) = COPY %r2
; CHECK-DAG: [[VREGP0HI:%[0-9]+]](s32) = COPY %r3
; LITTLE: [[VREGP0:%[0-9]+]](s64) = G_MERGE_VALUES [[VREGP0LO]](s32), [[VREGP0HI]](s32)
; BIG: [[VREGP0:%[0-9]+]](s64) = G_MERGE_VALUES [[VREGP0HI]](s32), [[VREGP0LO]](s32)
; CHECK: [[FIP1:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[P1]]
; CHECK: [[VREGP1:%[0-9]+]](s64) = G_LOAD [[FIP1]](p0){{.*}}load 8
; CHECK: [[VREGV:%[0-9]+]](s64) = G_FADD [[VREGP0]], [[VREGP1]]
; LITTLE: [[VREGVLO:%[0-9]+]](s32), [[VREGVHI:%[0-9]+]](s32) = G_UNMERGE_VALUES [[VREGV]](s64)
; BIG: [[VREGVHI:%[0-9]+]](s32), [[VREGVLO:%[0-9]+]](s32) = G_UNMERGE_VALUES [[VREGV]](s64)
; CHECK-DAG: %r0 = COPY [[VREGVLO]]
; CHECK-DAG: %r1 = COPY [[VREGVHI]]
; CHECK: BX_RET 14, _, implicit %r0, implicit %r1
entry:
  %v = fadd double %p0, %p1
  ret double %v
}

define arm_aapcscc double @test_double_gap2_aapcscc(double %p0, float %filler,
                                                    double %p1) {
; CHECK-LABEL: name: test_double_gap2_aapcscc
; CHECK: fixedStack:
; CHECK-DAG: id: [[P1:[0-9]+]]{{.*}}offset: 0{{.*}}size: 8
; CHECK: liveins: %r0, %r1, %r2
; CHECK-DAG: [[VREGP0LO:%[0-9]+]](s32) = COPY %r0
; CHECK-DAG: [[VREGP0HI:%[0-9]+]](s32) = COPY %r1
; LITTLE: [[VREGP0:%[0-9]+]](s64) = G_MERGE_VALUES [[VREGP0LO]](s32), [[VREGP0HI]](s32)
; BIG: [[VREGP0:%[0-9]+]](s64) = G_MERGE_VALUES [[VREGP0HI]](s32), [[VREGP0LO]](s32)
; CHECK: [[FIP1:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[P1]]
; CHECK: [[VREGP1:%[0-9]+]](s64) = G_LOAD [[FIP1]](p0){{.*}}load 8
; CHECK: [[VREGV:%[0-9]+]](s64) = G_FADD [[VREGP0]], [[VREGP1]]
; LITTLE: [[VREGVLO:%[0-9]+]](s32), [[VREGVHI:%[0-9]+]](s32) = G_UNMERGE_VALUES [[VREGV]](s64)
; BIG: [[VREGVHI:%[0-9]+]](s32), [[VREGVLO:%[0-9]+]](s32) = G_UNMERGE_VALUES [[VREGV]](s64)
; CHECK-DAG: %r0 = COPY [[VREGVLO]]
; CHECK-DAG: %r1 = COPY [[VREGVHI]]
; CHECK: BX_RET 14, _, implicit %r0, implicit %r1
entry:
  %v = fadd double %p0, %p1
  ret double %v
}

define arm_aapcscc void @test_indirect_call(void() *%fptr) {
; CHECK-LABEL: name: test_indirect_call
; CHECK: registers:
; CHECK-NEXT: id: [[FPTR:[0-9]+]], class: gpr
; CHECK: %[[FPTR]](p0) = COPY %r0
; CHECK: ADJCALLSTACKDOWN 0, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: BLX %[[FPTR]](p0), csr_aapcs, implicit-def %lr, implicit %sp
; CHECK: ADJCALLSTACKUP 0, 0, 14, _, implicit-def %sp, implicit %sp
entry:
  notail call arm_aapcscc void %fptr()
  ret void
}

declare arm_aapcscc void @call_target()

define arm_aapcscc void @test_direct_call() {
; CHECK-LABEL: name: test_direct_call
; CHECK: ADJCALLSTACKDOWN 0, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: BLX @call_target, csr_aapcs, implicit-def %lr, implicit %sp
; CHECK: ADJCALLSTACKUP 0, 0, 14, _, implicit-def %sp, implicit %sp
entry:
  notail call arm_aapcscc void @call_target()
  ret void
}

declare arm_aapcscc i32* @simple_reg_params_target(i32, i32*)

define arm_aapcscc i32* @test_call_simple_reg_params(i32 *%a, i32 %b) {
; CHECK-LABEL: name: test_call_simple_reg_params
; CHECK-DAG: [[AVREG:%[0-9]+]](p0) = COPY %r0
; CHECK-DAG: [[BVREG:%[0-9]+]](s32) = COPY %r1
; CHECK: ADJCALLSTACKDOWN 0, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK-DAG: %r0 = COPY [[BVREG]]
; CHECK-DAG: %r1 = COPY [[AVREG]]
; CHECK: BLX @simple_reg_params_target, csr_aapcs, implicit-def %lr, implicit %sp, implicit %r0, implicit %r1, implicit-def %r0
; CHECK: [[RVREG:%[0-9]+]](p0) = COPY %r0
; CHECK: ADJCALLSTACKUP 0, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: %r0 = COPY [[RVREG]]
; CHECK: BX_RET 14, _, implicit %r0
entry:
  %r = notail call arm_aapcscc i32 *@simple_reg_params_target(i32 %b, i32 *%a)
  ret i32 *%r
}

declare arm_aapcscc i32* @simple_stack_params_target(i32, i32*, i32, i32*, i32, i32*)

define arm_aapcscc i32* @test_call_simple_stack_params(i32 *%a, i32 %b) {
; CHECK-LABEL: name: test_call_simple_stack_params
; CHECK-DAG: [[AVREG:%[0-9]+]](p0) = COPY %r0
; CHECK-DAG: [[BVREG:%[0-9]+]](s32) = COPY %r1
; CHECK: ADJCALLSTACKDOWN 8, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK-DAG: %r0 = COPY [[BVREG]]
; CHECK-DAG: %r1 = COPY [[AVREG]]
; CHECK-DAG: %r2 = COPY [[BVREG]]
; CHECK-DAG: %r3 = COPY [[AVREG]]
; CHECK: [[SP1:%[0-9]+]](p0) = COPY %sp
; CHECK: [[OFF1:%[0-9]+]](s32) = G_CONSTANT i32 0
; CHECK: [[FI1:%[0-9]+]](p0) = G_GEP [[SP1]], [[OFF1]](s32)
; CHECK: G_STORE [[BVREG]](s32), [[FI1]](p0){{.*}}store 4
; CHECK: [[SP2:%[0-9]+]](p0) = COPY %sp
; CHECK: [[OFF2:%[0-9]+]](s32) = G_CONSTANT i32 4
; CHECK: [[FI2:%[0-9]+]](p0) = G_GEP [[SP2]], [[OFF2]](s32)
; CHECK: G_STORE [[AVREG]](p0), [[FI2]](p0){{.*}}store 4
; CHECK: BLX @simple_stack_params_target, csr_aapcs, implicit-def %lr, implicit %sp, implicit %r0, implicit %r1, implicit %r2, implicit %r3, implicit-def %r0
; CHECK: [[RVREG:%[0-9]+]](p0) = COPY %r0
; CHECK: ADJCALLSTACKUP 8, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: %r0 = COPY [[RVREG]]
; CHECK: BX_RET 14, _, implicit %r0
entry:
  %r = notail call arm_aapcscc i32 *@simple_stack_params_target(i32 %b, i32 *%a, i32 %b, i32 *%a, i32 %b, i32 *%a)
  ret i32 *%r
}

declare arm_aapcscc signext i16 @ext_target(i8 signext, i8 zeroext, i16 signext, i16 zeroext, i8 signext, i8 zeroext, i16 signext, i16 zeroext, i1 zeroext)

define arm_aapcscc signext i16 @test_call_ext_params(i8 %a, i16 %b, i1 %c) {
; CHECK-LABEL: name: test_call_ext_params
; CHECK-DAG: [[AVREG:%[0-9]+]](s8) = COPY %r0
; CHECK-DAG: [[BVREG:%[0-9]+]](s16) = COPY %r1
; CHECK-DAG: [[CVREG:%[0-9]+]](s1) = COPY %r2
; CHECK: ADJCALLSTACKDOWN 20, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: [[SEXTA:%[0-9]+]](s32) = G_SEXT [[AVREG]](s8)
; CHECK: %r0 = COPY [[SEXTA]]
; CHECK: [[ZEXTA:%[0-9]+]](s32) = G_ZEXT [[AVREG]](s8)
; CHECK: %r1 = COPY [[ZEXTA]]
; CHECK: [[SEXTB:%[0-9]+]](s32) = G_SEXT [[BVREG]](s16)
; CHECK: %r2 = COPY [[SEXTB]]
; CHECK: [[ZEXTB:%[0-9]+]](s32) = G_ZEXT [[BVREG]](s16)
; CHECK: %r3 = COPY [[ZEXTB]]
; CHECK: [[SP1:%[0-9]+]](p0) = COPY %sp
; CHECK: [[OFF1:%[0-9]+]](s32) = G_CONSTANT i32 0
; CHECK: [[FI1:%[0-9]+]](p0) = G_GEP [[SP1]], [[OFF1]](s32)
; CHECK: [[SEXTA2:%[0-9]+]](s32) = G_SEXT [[AVREG]]
; CHECK: G_STORE [[SEXTA2]](s32), [[FI1]](p0){{.*}}store 4
; CHECK: [[SP2:%[0-9]+]](p0) = COPY %sp
; CHECK: [[OFF2:%[0-9]+]](s32) = G_CONSTANT i32 4
; CHECK: [[FI2:%[0-9]+]](p0) = G_GEP [[SP2]], [[OFF2]](s32)
; CHECK: [[ZEXTA2:%[0-9]+]](s32) = G_ZEXT [[AVREG]]
; CHECK: G_STORE [[ZEXTA2]](s32), [[FI2]](p0){{.*}}store 4
; CHECK: [[SP3:%[0-9]+]](p0) = COPY %sp
; CHECK: [[OFF3:%[0-9]+]](s32) = G_CONSTANT i32 8
; CHECK: [[FI3:%[0-9]+]](p0) = G_GEP [[SP3]], [[OFF3]](s32)
; CHECK: [[SEXTB2:%[0-9]+]](s32) = G_SEXT [[BVREG]]
; CHECK: G_STORE [[SEXTB2]](s32), [[FI3]](p0){{.*}}store 4
; CHECK: [[SP4:%[0-9]+]](p0) = COPY %sp
; CHECK: [[OFF4:%[0-9]+]](s32) = G_CONSTANT i32 12
; CHECK: [[FI4:%[0-9]+]](p0) = G_GEP [[SP4]], [[OFF4]](s32)
; CHECK: [[ZEXTB2:%[0-9]+]](s32) = G_ZEXT [[BVREG]]
; CHECK: G_STORE [[ZEXTB2]](s32), [[FI4]](p0){{.*}}store 4
; CHECK: [[SP5:%[0-9]+]](p0) = COPY %sp
; CHECK: [[OFF5:%[0-9]+]](s32) = G_CONSTANT i32 16
; CHECK: [[FI5:%[0-9]+]](p0) = G_GEP [[SP5]], [[OFF5]](s32)
; CHECK: [[ZEXTC:%[0-9]+]](s32) = G_ZEXT [[CVREG]]
; CHECK: G_STORE [[ZEXTC]](s32), [[FI5]](p0){{.*}}store 4
; CHECK: BLX @ext_target, csr_aapcs, implicit-def %lr, implicit %sp, implicit %r0, implicit %r1, implicit %r2, implicit %r3, implicit-def %r0
; CHECK: [[RVREG:%[0-9]+]](s16) = COPY %r0
; CHECK: ADJCALLSTACKUP 20, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: [[RExtVREG:%[0-9]+]](s32) = G_SEXT [[RVREG]]
; CHECK: %r0 = COPY [[RExtVREG]]
; CHECK: BX_RET 14, _, implicit %r0
entry:
  %r = notail call arm_aapcscc signext i16 @ext_target(i8 signext %a, i8 zeroext %a, i16 signext %b, i16 zeroext %b, i8 signext %a, i8 zeroext %a, i16 signext %b, i16 zeroext %b, i1 zeroext %c)
  ret i16 %r
}

declare arm_aapcs_vfpcc double @vfpcc_fp_target(float, double)

define arm_aapcs_vfpcc double @test_call_vfpcc_fp_params(double %a, float %b) {
; CHECK-LABEL: name: test_call_vfpcc_fp_params
; CHECK-DAG: [[AVREG:%[0-9]+]](s64) = COPY %d0
; CHECK-DAG: [[BVREG:%[0-9]+]](s32) = COPY %s2
; CHECK: ADJCALLSTACKDOWN 0, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK-DAG: %s0 = COPY [[BVREG]]
; CHECK-DAG: %d1 = COPY [[AVREG]]
; CHECK: BLX @vfpcc_fp_target, csr_aapcs, implicit-def %lr, implicit %sp, implicit %s0, implicit %d1, implicit-def %d0
; CHECK: [[RVREG:%[0-9]+]](s64) = COPY %d0
; CHECK: ADJCALLSTACKUP 0, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: %d0 = COPY [[RVREG]]
; CHECK: BX_RET 14, _, implicit %d0
entry:
  %r = notail call arm_aapcs_vfpcc double @vfpcc_fp_target(float %b, double %a)
  ret double %r
}

declare arm_aapcscc double @aapcscc_fp_target(float, double, float, double)

define arm_aapcscc double @test_call_aapcs_fp_params(double %a, float %b) {
; CHECK-LABEL: name: test_call_aapcs_fp_params
; CHECK-DAG: [[A1:%[0-9]+]](s32) = COPY %r0
; CHECK-DAG: [[A2:%[0-9]+]](s32) = COPY %r1
; LITTLE-DAG: [[AVREG:%[0-9]+]](s64) = G_MERGE_VALUES [[A1]](s32), [[A2]](s32)
; BIG-DAG: [[AVREG:%[0-9]+]](s64) = G_MERGE_VALUES [[A2]](s32), [[A1]](s32)
; CHECK-DAG: [[BVREG:%[0-9]+]](s32) = COPY %r2
; CHECK: ADJCALLSTACKDOWN 16, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK-DAG: %r0 = COPY [[BVREG]]
; CHECK-DAG: [[A1:%[0-9]+]](s32), [[A2:%[0-9]+]](s32) = G_UNMERGE_VALUES [[AVREG]](s64)
; LITTLE-DAG: %r2 = COPY [[A1]]
; LITTLE-DAG: %r3 = COPY [[A2]]
; BIG-DAG: %r2 = COPY [[A2]]
; BIG-DAG: %r3 = COPY [[A1]]
; CHECK: [[SP1:%[0-9]+]](p0) = COPY %sp
; CHECK: [[OFF1:%[0-9]+]](s32) = G_CONSTANT i32 0
; CHECK: [[FI1:%[0-9]+]](p0) = G_GEP [[SP1]], [[OFF1]](s32)
; CHECK: G_STORE [[BVREG]](s32), [[FI1]](p0){{.*}}store 4
; CHECK: [[SP2:%[0-9]+]](p0) = COPY %sp
; CHECK: [[OFF2:%[0-9]+]](s32) = G_CONSTANT i32 8
; CHECK: [[FI2:%[0-9]+]](p0) = G_GEP [[SP2]], [[OFF2]](s32)
; CHECK: G_STORE [[AVREG]](s64), [[FI2]](p0){{.*}}store 8
; CHECK: BLX @aapcscc_fp_target, csr_aapcs, implicit-def %lr, implicit %sp, implicit %r0, implicit %r2, implicit %r3, implicit-def %r0, implicit-def %r1
; CHECK-DAG: [[R1:%[0-9]+]](s32) = COPY %r0
; CHECK-DAG: [[R2:%[0-9]+]](s32) = COPY %r1
; LITTLE: [[RVREG:%[0-9]+]](s64) = G_MERGE_VALUES [[R1]](s32), [[R2]](s32)
; BIG: [[RVREG:%[0-9]+]](s64) = G_MERGE_VALUES [[R2]](s32), [[R1]](s32)
; CHECK: ADJCALLSTACKUP 16, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: [[R1:%[0-9]+]](s32), [[R2:%[0-9]+]](s32) = G_UNMERGE_VALUES [[RVREG]](s64)
; LITTLE-DAG: %r0 = COPY [[R1]]
; LITTLE-DAG: %r1 = COPY [[R2]]
; BIG-DAG: %r0 = COPY [[R2]]
; BIG-DAG: %r1 = COPY [[R1]]
; CHECK: BX_RET 14, _, implicit %r0, implicit %r1
entry:
  %r = notail call arm_aapcscc double @aapcscc_fp_target(float %b, double %a, float %b, double %a)
  ret double %r
}

declare arm_aapcscc float @different_call_conv_target(float)

define arm_aapcs_vfpcc float @test_call_different_call_conv(float %x) {
; CHECK-LABEL: name: test_call_different_call_conv
; CHECK: [[X:%[0-9]+]](s32) = COPY %s0
; CHECK: ADJCALLSTACKDOWN 0, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: %r0 = COPY [[X]]
; CHECK: BLX @different_call_conv_target, csr_aapcs, implicit-def %lr, implicit %sp, implicit %r0, implicit-def %r0
; CHECK: [[R:%[0-9]+]](s32) = COPY %r0
; CHECK: ADJCALLSTACKUP 0, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: %s0 = COPY [[R]]
; CHECK: BX_RET 14, _, implicit %s0
entry:
  %r = notail call arm_aapcscc float @different_call_conv_target(float %x)
  ret float %r
}

declare arm_aapcscc [3 x i32] @tiny_int_arrays_target([2 x i32])

define arm_aapcscc [3 x i32] @test_tiny_int_arrays([2 x i32] %arr) {
; CHECK-LABEL: name: test_tiny_int_arrays
; CHECK: liveins: %r0, %r1
; CHECK: [[R0:%[0-9]+]](s32) = COPY %r0
; CHECK: [[R1:%[0-9]+]](s32) = COPY %r1
; CHECK: [[ARG_ARR:%[0-9]+]](s64) = G_MERGE_VALUES [[R0]](s32), [[R1]](s32)
; CHECK: ADJCALLSTACKDOWN 0, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: [[R0:%[0-9]+]](s32), [[R1:%[0-9]+]](s32) = G_UNMERGE_VALUES [[ARG_ARR]](s64)
; CHECK: %r0 = COPY [[R0]]
; CHECK: %r1 = COPY [[R1]]
; CHECK: BLX @tiny_int_arrays_target, csr_aapcs, implicit-def %lr, implicit %sp, implicit %r0, implicit %r1, implicit-def %r0, implicit-def %r1
; CHECK: [[R0:%[0-9]+]](s32) = COPY %r0
; CHECK: [[R1:%[0-9]+]](s32) = COPY %r1
; CHECK: [[R2:%[0-9]+]](s32) = COPY %r2
; CHECK: [[RES_ARR:%[0-9]+]](s96) = G_MERGE_VALUES [[R0]](s32), [[R1]](s32), [[R2]](s32)
; CHECK: ADJCALLSTACKUP 0, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: [[R0:%[0-9]+]](s32), [[R1:%[0-9]+]](s32), [[R2:%[0-9]+]](s32) = G_UNMERGE_VALUES [[RES_ARR]](s96)
; FIXME: This doesn't seem correct with regard to the AAPCS docs (which say
; that composite types larger than 4 bytes should be passed through memory),
; but it's what DAGISel does. We should fix it in the common code for both.
; CHECK: %r0 = COPY [[R0]]
; CHECK: %r1 = COPY [[R1]]
; CHECK: %r2 = COPY [[R2]]
; CHECK: BX_RET 14, _, implicit %r0, implicit %r1, implicit %r2
entry:
  %r = notail call arm_aapcscc [3 x i32] @tiny_int_arrays_target([2 x i32] %arr)
  ret [3 x i32] %r
}

declare arm_aapcscc void @multiple_int_arrays_target([2 x i32], [2 x i32])

define arm_aapcscc void @test_multiple_int_arrays([2 x i32] %arr0, [2 x i32] %arr1) {
; CHECK-LABEL: name: test_multiple_int_arrays
; CHECK: liveins: %r0, %r1
; CHECK: [[R0:%[0-9]+]](s32) = COPY %r0
; CHECK: [[R1:%[0-9]+]](s32) = COPY %r1
; CHECK: [[R2:%[0-9]+]](s32) = COPY %r2
; CHECK: [[R3:%[0-9]+]](s32) = COPY %r3
; CHECK: [[ARG_ARR0:%[0-9]+]](s64) = G_MERGE_VALUES [[R0]](s32), [[R1]](s32)
; CHECK: [[ARG_ARR1:%[0-9]+]](s64) = G_MERGE_VALUES [[R2]](s32), [[R3]](s32)
; CHECK: ADJCALLSTACKDOWN 0, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: [[R0:%[0-9]+]](s32), [[R1:%[0-9]+]](s32) = G_UNMERGE_VALUES [[ARG_ARR0]](s64)
; CHECK: [[R2:%[0-9]+]](s32), [[R3:%[0-9]+]](s32) = G_UNMERGE_VALUES [[ARG_ARR1]](s64)
; CHECK: %r0 = COPY [[R0]]
; CHECK: %r1 = COPY [[R1]]
; CHECK: %r2 = COPY [[R2]]
; CHECK: %r3 = COPY [[R3]]
; CHECK: BLX @multiple_int_arrays_target, csr_aapcs, implicit-def %lr, implicit %sp, implicit %r0, implicit %r1, implicit %r2, implicit %r3
; CHECK: ADJCALLSTACKUP 0, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: BX_RET 14, _
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
; CHECK: liveins: %r0, %r1, %r2, %r3
; CHECK-DAG: [[R0:%[0-9]+]](s32) = COPY %r0
; CHECK-DAG: [[R1:%[0-9]+]](s32) = COPY %r1
; CHECK-DAG: [[R2:%[0-9]+]](s32) = COPY %r2
; CHECK-DAG: [[R3:%[0-9]+]](s32) = COPY %r3
; CHECK: [[FIRST_STACK_ELEMENT_FI:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[FIRST_STACK_ID]]
; CHECK: [[FIRST_STACK_ELEMENT:%[0-9]+]](s32) = G_LOAD [[FIRST_STACK_ELEMENT_FI]]{{.*}}load 4 from %fixed-stack.[[FIRST_STACK_ID]]
; CHECK: [[LAST_STACK_ELEMENT_FI:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[LAST_STACK_ID]]
; CHECK: [[LAST_STACK_ELEMENT:%[0-9]+]](s32) = G_LOAD [[LAST_STACK_ELEMENT_FI]]{{.*}}load 4 from %fixed-stack.[[LAST_STACK_ID]]
; CHECK: [[ARG_ARR:%[0-9]+]](s640) = G_MERGE_VALUES [[R0]](s32), [[R1]](s32), [[R2]](s32), [[R3]](s32), [[FIRST_STACK_ELEMENT]](s32), {{.*}}, [[LAST_STACK_ELEMENT]](s32)
; CHECK: ADJCALLSTACKDOWN 64, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: [[R0:%[0-9]+]](s32), [[R1:%[0-9]+]](s32), [[R2:%[0-9]+]](s32), [[R3:%[0-9]+]](s32), [[FIRST_STACK_ELEMENT:%[0-9]+]](s32), {{.*}}, [[LAST_STACK_ELEMENT:%[0-9]+]](s32) = G_UNMERGE_VALUES [[ARG_ARR]](s640)
; CHECK: %r0 = COPY [[R0]]
; CHECK: %r1 = COPY [[R1]]
; CHECK: %r2 = COPY [[R2]]
; CHECK: %r3 = COPY [[R3]]
; CHECK: [[SP:%[0-9]+]](p0) = COPY %sp
; CHECK: [[OFF_FIRST_ELEMENT:%[0-9]+]](s32) = G_CONSTANT i32 0
; CHECK: [[FIRST_STACK_ARG_ADDR:%[0-9]+]](p0) = G_GEP [[SP]], [[OFF_FIRST_ELEMENT]](s32)
; CHECK: G_STORE [[FIRST_STACK_ELEMENT]](s32), [[FIRST_STACK_ARG_ADDR]]{{.*}}store 4
; Match the second-to-last offset, so we can get the correct SP for the last element
; CHECK: G_CONSTANT i32 56
; CHECK: [[SP:%[0-9]+]](p0) = COPY %sp
; CHECK: [[OFF_LAST_ELEMENT:%[0-9]+]](s32) = G_CONSTANT i32 60
; CHECK: [[LAST_STACK_ARG_ADDR:%[0-9]+]](p0) = G_GEP [[SP]], [[OFF_LAST_ELEMENT]](s32)
; CHECK: G_STORE [[LAST_STACK_ELEMENT]](s32), [[LAST_STACK_ARG_ADDR]]{{.*}}store 4
; CHECK: BLX @large_int_arrays_target, csr_aapcs, implicit-def %lr, implicit %sp, implicit %r0, implicit %r1, implicit %r2, implicit %r3
; CHECK: ADJCALLSTACKUP 64, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: BX_RET 14, _
entry:
  notail call arm_aapcscc void @large_int_arrays_target([20 x i32] %arr)
  ret void
}

declare arm_aapcscc [2 x float] @fp_arrays_aapcs_target([3 x double])

define arm_aapcscc [2 x float] @test_fp_arrays_aapcs([3 x double] %arr) {
; CHECK-LABEL: name: test_fp_arrays_aapcs
; CHECK: fixedStack:
; CHECK: id: [[ARR2_ID:[0-9]+]], type: default, offset: 0, size: 8,
; CHECK: liveins: %r0, %r1, %r2, %r3
; CHECK: [[ARR0_0:%[0-9]+]](s32) = COPY %r0
; CHECK: [[ARR0_1:%[0-9]+]](s32) = COPY %r1
; LITTLE: [[ARR0:%[0-9]+]](s64) = G_MERGE_VALUES [[ARR0_0]](s32), [[ARR0_1]](s32)
; BIG: [[ARR0:%[0-9]+]](s64) = G_MERGE_VALUES [[ARR0_1]](s32), [[ARR0_0]](s32)
; CHECK: [[ARR1_0:%[0-9]+]](s32) = COPY %r2
; CHECK: [[ARR1_1:%[0-9]+]](s32) = COPY %r3
; LITTLE: [[ARR1:%[0-9]+]](s64) = G_MERGE_VALUES [[ARR1_0]](s32), [[ARR1_1]](s32)
; BIG: [[ARR1:%[0-9]+]](s64) = G_MERGE_VALUES [[ARR1_1]](s32), [[ARR1_0]](s32)
; CHECK: [[ARR2_FI:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[ARR2_ID]]
; CHECK: [[ARR2:%[0-9]+]](s64) = G_LOAD [[ARR2_FI]]{{.*}}load 8 from %fixed-stack.[[ARR2_ID]]
; CHECK: [[ARR_MERGED:%[0-9]+]](s192) = G_MERGE_VALUES [[ARR0]](s64), [[ARR1]](s64), [[ARR2]](s64)
; CHECK: ADJCALLSTACKDOWN 8, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: [[ARR0:%[0-9]+]](s64), [[ARR1:%[0-9]+]](s64), [[ARR2:%[0-9]+]](s64) = G_UNMERGE_VALUES [[ARR_MERGED]](s192)
; CHECK: [[ARR0_0:%[0-9]+]](s32), [[ARR0_1:%[0-9]+]](s32) = G_UNMERGE_VALUES [[ARR0]](s64)
; LITTLE: %r0 = COPY [[ARR0_0]](s32)
; LITTLE: %r1 = COPY [[ARR0_1]](s32)
; BIG: %r0 = COPY [[ARR0_1]](s32)
; BIG: %r1 = COPY [[ARR0_0]](s32)
; CHECK: [[ARR1_0:%[0-9]+]](s32), [[ARR1_1:%[0-9]+]](s32) = G_UNMERGE_VALUES [[ARR1]](s64)
; LITTLE: %r2 = COPY [[ARR1_0]](s32)
; LITTLE: %r3 = COPY [[ARR1_1]](s32)
; BIG: %r2 = COPY [[ARR1_1]](s32)
; BIG: %r3 = COPY [[ARR1_0]](s32)
; CHECK: [[SP:%[0-9]+]](p0) = COPY %sp
; CHECK: [[ARR2_OFFSET:%[0-9]+]](s32) = G_CONSTANT i32 0
; CHECK: [[ARR2_ADDR:%[0-9]+]](p0) = G_GEP [[SP]], [[ARR2_OFFSET]](s32)
; CHECK: G_STORE [[ARR2]](s64), [[ARR2_ADDR]](p0){{.*}}store 8
; CHECK: BLX @fp_arrays_aapcs_target, csr_aapcs, implicit-def %lr, implicit %sp, implicit %r0, implicit %r1, implicit %r2, implicit %r3, implicit-def %r0, implicit-def %r1
; CHECK: [[R0:%[0-9]+]](s32) = COPY %r0
; CHECK: [[R1:%[0-9]+]](s32) = COPY %r1
; CHECK: [[R_MERGED:%[0-9]+]](s64) = G_MERGE_VALUES [[R0]](s32), [[R1]](s32)
; CHECK: ADJCALLSTACKUP 8, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: [[R0:%[0-9]+]](s32), [[R1:%[0-9]+]](s32) = G_UNMERGE_VALUES [[R_MERGED]](s64)
; CHECK: %r0 = COPY [[R0]]
; CHECK: %r1 = COPY [[R1]]
; CHECK: BX_RET 14, _, implicit %r0, implicit %r1
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
; CHECK: liveins: %d0, %d1, %d2, %s6, %s7, %s8
; CHECK: [[X0:%[0-9]+]](s64) = COPY %d0
; CHECK: [[X1:%[0-9]+]](s64) = COPY %d1
; CHECK: [[X2:%[0-9]+]](s64) = COPY %d2
; CHECK: [[Y0:%[0-9]+]](s32) = COPY %s6
; CHECK: [[Y1:%[0-9]+]](s32) = COPY %s7
; CHECK: [[Y2:%[0-9]+]](s32) = COPY %s8
; CHECK: [[Z0_FI:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[Z0_ID]]
; CHECK: [[Z0:%[0-9]+]](s64) = G_LOAD [[Z0_FI]]{{.*}}load 8
; CHECK: [[Z1_FI:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[Z1_ID]]
; CHECK: [[Z1:%[0-9]+]](s64) = G_LOAD [[Z1_FI]]{{.*}}load 8
; CHECK: [[Z2_FI:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[Z2_ID]]
; CHECK: [[Z2:%[0-9]+]](s64) = G_LOAD [[Z2_FI]]{{.*}}load 8
; CHECK: [[Z3_FI:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[Z3_ID]]
; CHECK: [[Z3:%[0-9]+]](s64) = G_LOAD [[Z3_FI]]{{.*}}load 8
; CHECK: [[X_ARR:%[0-9]+]](s192) = G_MERGE_VALUES [[X0]](s64), [[X1]](s64), [[X2]](s64)
; CHECK: [[Y_ARR:%[0-9]+]](s96) = G_MERGE_VALUES [[Y0]](s32), [[Y1]](s32), [[Y2]](s32)
; CHECK: [[Z_ARR:%[0-9]+]](s256) = G_MERGE_VALUES [[Z0]](s64), [[Z1]](s64), [[Z2]](s64), [[Z3]](s64)
; CHECK: ADJCALLSTACKDOWN 32, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: [[X0:%[0-9]+]](s64), [[X1:%[0-9]+]](s64), [[X2:%[0-9]+]](s64) = G_UNMERGE_VALUES [[X_ARR]](s192)
; CHECK: [[Y0:%[0-9]+]](s32), [[Y1:%[0-9]+]](s32), [[Y2:%[0-9]+]](s32) = G_UNMERGE_VALUES [[Y_ARR]](s96)
; CHECK: [[Z0:%[0-9]+]](s64), [[Z1:%[0-9]+]](s64), [[Z2:%[0-9]+]](s64), [[Z3:%[0-9]+]](s64) = G_UNMERGE_VALUES [[Z_ARR]](s256)
; CHECK: %d0 = COPY [[X0]](s64)
; CHECK: %d1 = COPY [[X1]](s64)
; CHECK: %d2 = COPY [[X2]](s64)
; CHECK: %s6 = COPY [[Y0]](s32)
; CHECK: %s7 = COPY [[Y1]](s32)
; CHECK: %s8 = COPY [[Y2]](s32)
; CHECK: [[SP:%[0-9]+]](p0) = COPY %sp
; CHECK: [[Z0_OFFSET:%[0-9]+]](s32) = G_CONSTANT i32 0
; CHECK: [[Z0_ADDR:%[0-9]+]](p0) = G_GEP [[SP]], [[Z0_OFFSET]](s32)
; CHECK: G_STORE [[Z0]](s64), [[Z0_ADDR]](p0){{.*}}store 8
; CHECK: [[SP:%[0-9]+]](p0) = COPY %sp
; CHECK: [[Z1_OFFSET:%[0-9]+]](s32) = G_CONSTANT i32 8
; CHECK: [[Z1_ADDR:%[0-9]+]](p0) = G_GEP [[SP]], [[Z1_OFFSET]](s32)
; CHECK: G_STORE [[Z1]](s64), [[Z1_ADDR]](p0){{.*}}store 8
; CHECK: [[SP:%[0-9]+]](p0) = COPY %sp
; CHECK: [[Z2_OFFSET:%[0-9]+]](s32) = G_CONSTANT i32 16
; CHECK: [[Z2_ADDR:%[0-9]+]](p0) = G_GEP [[SP]], [[Z2_OFFSET]](s32)
; CHECK: G_STORE [[Z2]](s64), [[Z2_ADDR]](p0){{.*}}store 8
; CHECK: [[SP:%[0-9]+]](p0) = COPY %sp
; CHECK: [[Z3_OFFSET:%[0-9]+]](s32) = G_CONSTANT i32 24
; CHECK: [[Z3_ADDR:%[0-9]+]](p0) = G_GEP [[SP]], [[Z3_OFFSET]](s32)
; CHECK: G_STORE [[Z3]](s64), [[Z3_ADDR]](p0){{.*}}store 8
; CHECK: BLX @fp_arrays_aapcs_vfp_target, csr_aapcs, implicit-def %lr, implicit %sp, implicit %d0, implicit %d1, implicit %d2, implicit %s6, implicit %s7, implicit %s8, implicit-def %s0, implicit-def %s1, implicit-def %s2, implicit-def %s3
; CHECK: [[R0:%[0-9]+]](s32) = COPY %s0
; CHECK: [[R1:%[0-9]+]](s32) = COPY %s1
; CHECK: [[R2:%[0-9]+]](s32) = COPY %s2
; CHECK: [[R3:%[0-9]+]](s32) = COPY %s3
; CHECK: [[R_MERGED:%[0-9]+]](s128) = G_MERGE_VALUES [[R0]](s32), [[R1]](s32), [[R2]](s32), [[R3]](s32)
; CHECK: ADJCALLSTACKUP 32, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: [[R0:%[0-9]+]](s32), [[R1:%[0-9]+]](s32), [[R2:%[0-9]+]](s32), [[R3:%[0-9]+]](s32) = G_UNMERGE_VALUES [[R_MERGED]](s128)
; CHECK: %s0 = COPY [[R0]]
; CHECK: %s1 = COPY [[R1]]
; CHECK: %s2 = COPY [[R2]]
; CHECK: %s3 = COPY [[R3]]
; CHECK: BX_RET 14, _, implicit %s0, implicit %s1, implicit %s2, implicit %s3
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
; CHECK: liveins: %r0, %r1, %r2, %r3
; CHECK-DAG: [[R0:%[0-9]+]](s32) = COPY %r0
; CHECK-DAG: [[R1:%[0-9]+]](s32) = COPY %r1
; CHECK-DAG: [[R2:%[0-9]+]](s32) = COPY %r2
; CHECK-DAG: [[R3:%[0-9]+]](s32) = COPY %r3
; CHECK: [[FIRST_STACK_ELEMENT_FI:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[FIRST_STACK_ID]]
; CHECK: [[FIRST_STACK_ELEMENT:%[0-9]+]](s32) = G_LOAD [[FIRST_STACK_ELEMENT_FI]]{{.*}}load 4 from %fixed-stack.[[FIRST_STACK_ID]]
; CHECK: [[LAST_STACK_ELEMENT_FI:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[LAST_STACK_ID]]
; CHECK: [[LAST_STACK_ELEMENT:%[0-9]+]](s32) = G_LOAD [[LAST_STACK_ELEMENT_FI]]{{.*}}load 4 from %fixed-stack.[[LAST_STACK_ID]]
; CHECK: [[ARG_ARR:%[0-9]+]](s768) = G_MERGE_VALUES [[R0]](s32), [[R1]](s32), [[R2]](s32), [[R3]](s32), [[FIRST_STACK_ELEMENT]](s32), {{.*}}, [[LAST_STACK_ELEMENT]](s32)
; CHECK: ADJCALLSTACKDOWN 80, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: [[R0:%[0-9]+]](s32), [[R1:%[0-9]+]](s32), [[R2:%[0-9]+]](s32), [[R3:%[0-9]+]](s32), [[FIRST_STACK_ELEMENT:%[0-9]+]](s32), {{.*}}, [[LAST_STACK_ELEMENT:%[0-9]+]](s32) = G_UNMERGE_VALUES [[ARG_ARR]](s768)
; CHECK: %r0 = COPY [[R0]]
; CHECK: %r1 = COPY [[R1]]
; CHECK: %r2 = COPY [[R2]]
; CHECK: %r3 = COPY [[R3]]
; CHECK: [[SP:%[0-9]+]](p0) = COPY %sp
; CHECK: [[OFF_FIRST_ELEMENT:%[0-9]+]](s32) = G_CONSTANT i32 0
; CHECK: [[FIRST_STACK_ARG_ADDR:%[0-9]+]](p0) = G_GEP [[SP]], [[OFF_FIRST_ELEMENT]](s32)
; CHECK: G_STORE [[FIRST_STACK_ELEMENT]](s32), [[FIRST_STACK_ARG_ADDR]]{{.*}}store 4
; Match the second-to-last offset, so we can get the correct SP for the last element
; CHECK: G_CONSTANT i32 72
; CHECK: [[SP:%[0-9]+]](p0) = COPY %sp
; CHECK: [[OFF_LAST_ELEMENT:%[0-9]+]](s32) = G_CONSTANT i32 76
; CHECK: [[LAST_STACK_ARG_ADDR:%[0-9]+]](p0) = G_GEP [[SP]], [[OFF_LAST_ELEMENT]](s32)
; CHECK: G_STORE [[LAST_STACK_ELEMENT]](s32), [[LAST_STACK_ARG_ADDR]]{{.*}}store 4
; CHECK: BLX @tough_arrays_target, csr_aapcs, implicit-def %lr, implicit %sp, implicit %r0, implicit %r1, implicit %r2, implicit %r3, implicit-def %r0, implicit-def %r1
; CHECK: [[R0:%[0-9]+]](s32) = COPY %r0
; CHECK: [[R1:%[0-9]+]](s32) = COPY %r1
; CHECK: [[RES_ARR:%[0-9]+]](s64) = G_MERGE_VALUES [[R0]](s32), [[R1]](s32)
; CHECK: ADJCALLSTACKUP 80, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: [[R0:%[0-9]+]](s32), [[R1:%[0-9]+]](s32) = G_UNMERGE_VALUES [[RES_ARR]](s64)
; CHECK: %r0 = COPY [[R0]]
; CHECK: %r1 = COPY [[R1]]
; CHECK: BX_RET 14, _, implicit %r0, implicit %r1
entry:
  %r = notail call arm_aapcscc [2 x i32*] @tough_arrays_target([6 x [4 x i32]] %arr)
  ret [2 x i32*] %r
}

declare arm_aapcscc {i32, i32} @structs_target({i32, i32})

define arm_aapcscc {i32, i32} @test_structs({i32, i32} %x) {
; CHECK-LABEL: test_structs
; CHECK: liveins: %r0, %r1
; CHECK-DAG: [[X0:%[0-9]+]](s32) = COPY %r0
; CHECK-DAG: [[X1:%[0-9]+]](s32) = COPY %r1
; CHECK: [[X:%[0-9]+]](s64) = G_MERGE_VALUES [[X0]](s32), [[X1]](s32)
; CHECK: ADJCALLSTACKDOWN 0, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: [[X0:%[0-9]+]](s32), [[X1:%[0-9]+]](s32) = G_UNMERGE_VALUES [[X]](s64)
; CHECK-DAG: %r0 = COPY [[X0]](s32)
; CHECK-DAG: %r1 = COPY [[X1]](s32)
; CHECK: BLX @structs_target, csr_aapcs, implicit-def %lr, implicit %sp, implicit %r0, implicit %r1, implicit-def %r0, implicit-def %r1
; CHECK: [[R0:%[0-9]+]](s32) = COPY %r0
; CHECK: [[R1:%[0-9]+]](s32) = COPY %r1
; CHECK: [[R:%[0-9]+]](s64) = G_MERGE_VALUES [[R0]](s32), [[R1]](s32)
; CHECK: ADJCALLSTACKUP 0, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: [[R0:%[0-9]+]](s32), [[R1:%[0-9]+]](s32) = G_UNMERGE_VALUES [[R]](s64)
; CHECK: %r0 = COPY [[R0]](s32)
; CHECK: %r1 = COPY [[R1]](s32)
; CHECK: BX_RET 14, _, implicit %r0, implicit %r1
  %r = notail call arm_aapcscc {i32, i32} @structs_target({i32, i32} %x)
  ret {i32, i32} %r
}

define i32 @test_shufflevector_s32_v2s32(i32 %arg) {
; CHECK-LABEL: name: test_shufflevector_s32_v2s32
; CHECK: [[ARG:%[0-9]+]](s32) = COPY %r0
; CHECK-DAG: [[UNDEF:%[0-9]+]](s32) = G_IMPLICIT_DEF
; CHECK-DAG: [[C0:%[0-9]+]](s32) = G_CONSTANT i32 0
; CHECK-DAG: [[MASK:%[0-9]+]](<2 x s32>) = G_MERGE_VALUES [[C0]](s32), [[C0]](s32)
; CHECK: [[VEC:%[0-9]+]](<2 x s32>) = G_SHUFFLE_VECTOR [[ARG]](s32), [[UNDEF]], [[MASK]](<2 x s32>)
; CHECK: G_EXTRACT_VECTOR_ELT [[VEC]](<2 x s32>)
  %vec = insertelement <1 x i32> undef, i32 %arg, i32 0
  %shuffle = shufflevector <1 x i32> %vec, <1 x i32> undef, <2 x i32> zeroinitializer
  %res = extractelement <2 x i32> %shuffle, i32 0
  ret i32 %res
}

define i32 @test_shufflevector_v2s32_v3s32(i32 %arg1, i32 %arg2) {
; CHECK-LABEL: name: test_shufflevector_v2s32_v3s32
; CHECK: [[ARG1:%[0-9]+]](s32) = COPY %r0
; CHECK: [[ARG2:%[0-9]+]](s32) = COPY %r1
; CHECK-DAG: [[UNDEF:%[0-9]+]](<2 x s32>) = G_IMPLICIT_DEF
; CHECK-DAG: [[C0:%[0-9]+]](s32) = G_CONSTANT i32 0
; CHECK-DAG: [[C1:%[0-9]+]](s32) = G_CONSTANT i32 1
; CHECK-DAG: [[MASK:%[0-9]+]](<3 x s32>) = G_MERGE_VALUES [[C1]](s32), [[C0]](s32), [[C1]](s32)
; CHECK-DAG: [[V1:%[0-9]+]](<2 x s32>) = G_INSERT_VECTOR_ELT [[UNDEF]], [[ARG1]](s32), [[C0]](s32)
; CHECK-DAG: [[V2:%[0-9]+]](<2 x s32>) = G_INSERT_VECTOR_ELT [[V1]], [[ARG2]](s32), [[C1]](s32)
; CHECK: [[VEC:%[0-9]+]](<3 x s32>) = G_SHUFFLE_VECTOR [[V2]](<2 x s32>), [[UNDEF]], [[MASK]](<3 x s32>)
; CHECK: G_EXTRACT_VECTOR_ELT [[VEC]](<3 x s32>)
  %v1 = insertelement <2 x i32> undef, i32 %arg1, i32 0
  %v2 = insertelement <2 x i32> %v1, i32 %arg2, i32 1
  %shuffle = shufflevector <2 x i32> %v2, <2 x i32> undef, <3 x i32> <i32 1, i32 0, i32 1>
  %res = extractelement <3 x i32> %shuffle, i32 0
  ret i32 %res
}


define i32 @test_shufflevector_v2s32_v4s32(i32 %arg1, i32 %arg2) {
; CHECK-LABEL: name: test_shufflevector_v2s32_v4s32
; CHECK: [[ARG1:%[0-9]+]](s32) = COPY %r0
; CHECK: [[ARG2:%[0-9]+]](s32) = COPY %r1
; CHECK-DAG: [[UNDEF:%[0-9]+]](<2 x s32>) = G_IMPLICIT_DEF
; CHECK-DAG: [[C0:%[0-9]+]](s32) = G_CONSTANT i32 0
; CHECK-DAG: [[C1:%[0-9]+]](s32) = G_CONSTANT i32 1
; CHECK-DAG: [[MASK:%[0-9]+]](<4 x s32>) = G_MERGE_VALUES [[C0]](s32), [[C0]](s32), [[C0]](s32), [[C0]](s32)
; CHECK-DAG: [[V1:%[0-9]+]](<2 x s32>) = G_INSERT_VECTOR_ELT [[UNDEF]], [[ARG1]](s32), [[C0]](s32)
; CHECK-DAG: [[V2:%[0-9]+]](<2 x s32>) = G_INSERT_VECTOR_ELT [[V1]], [[ARG2]](s32), [[C1]](s32)
; CHECK: [[VEC:%[0-9]+]](<4 x s32>) = G_SHUFFLE_VECTOR [[V2]](<2 x s32>), [[UNDEF]], [[MASK]](<4 x s32>)
; CHECK: G_EXTRACT_VECTOR_ELT [[VEC]](<4 x s32>)
  %v1 = insertelement <2 x i32> undef, i32 %arg1, i32 0
  %v2 = insertelement <2 x i32> %v1, i32 %arg2, i32 1
  %shuffle = shufflevector <2 x i32> %v2, <2 x i32> undef, <4 x i32> zeroinitializer
  %res = extractelement <4 x i32> %shuffle, i32 0
  ret i32 %res
}

define i32 @test_shufflevector_v4s32_v2s32(i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4) {
; CHECK-LABEL: name: test_shufflevector_v4s32_v2s32
; CHECK: [[ARG1:%[0-9]+]](s32) = COPY %r0
; CHECK: [[ARG2:%[0-9]+]](s32) = COPY %r1
; CHECK: [[ARG3:%[0-9]+]](s32) = COPY %r2
; CHECK: [[ARG4:%[0-9]+]](s32) = COPY %r3
; CHECK-DAG: [[UNDEF:%[0-9]+]](<4 x s32>) = G_IMPLICIT_DEF
; CHECK-DAG: [[C0:%[0-9]+]](s32) = G_CONSTANT i32 0
; CHECK-DAG: [[C1:%[0-9]+]](s32) = G_CONSTANT i32 1
; CHECK-DAG: [[C2:%[0-9]+]](s32) = G_CONSTANT i32 2
; CHECK-DAG: [[C3:%[0-9]+]](s32) = G_CONSTANT i32 3
; CHECK-DAG: [[MASK:%[0-9]+]](<2 x s32>) = G_MERGE_VALUES [[C1]](s32), [[C3]](s32)
; CHECK-DAG: [[V1:%[0-9]+]](<4 x s32>) = G_INSERT_VECTOR_ELT [[UNDEF]], [[ARG1]](s32), [[C0]](s32)
; CHECK-DAG: [[V2:%[0-9]+]](<4 x s32>) = G_INSERT_VECTOR_ELT [[V1]], [[ARG2]](s32), [[C1]](s32)
; CHECK-DAG: [[V3:%[0-9]+]](<4 x s32>) = G_INSERT_VECTOR_ELT [[V2]], [[ARG3]](s32), [[C2]](s32)
; CHECK-DAG: [[V4:%[0-9]+]](<4 x s32>) = G_INSERT_VECTOR_ELT [[V3]], [[ARG4]](s32), [[C3]](s32)
; CHECK: [[VEC:%[0-9]+]](<2 x s32>) = G_SHUFFLE_VECTOR [[V4]](<4 x s32>), [[UNDEF]], [[MASK]](<2 x s32>)
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
; CHECK: [[C1:%[0-9]+]](s32) = G_CONSTANT i32 1
; CHECK: [[C2:%[0-9]+]](s32) = G_CONSTANT i32 2
; CHECK: [[VEC:%[0-9]+]](<2 x s32>) = G_MERGE_VALUES [[C1]](s32), [[C2]](s32)
; CHECK: G_EXTRACT_VECTOR_ELT [[VEC]](<2 x s32>)
  %vec = extractvalue %struct.v2s32 {<2 x i32><i32 1, i32 2>}, 0
  %elt = extractelement <2 x i32> %vec, i32 0
  ret i32 %elt
}

%struct.v2s32.s32.s32 = type { <2 x i32>, i32, i32 }

define i32 @test_constantstruct_v2s32_s32_s32() {
; CHECK-LABEL: name: test_constantstruct_v2s32_s32_s32
; CHECK: [[C1:%[0-9]+]](s32) = G_CONSTANT i32 1
; CHECK: [[C2:%[0-9]+]](s32) = G_CONSTANT i32 2
; CHECK: [[VEC:%[0-9]+]](<2 x s32>) = G_MERGE_VALUES [[C1]](s32), [[C2]](s32)
; CHECK: [[C3:%[0-9]+]](s32) = G_CONSTANT i32 3
; CHECK: [[C4:%[0-9]+]](s32) = G_CONSTANT i32 4
; CHECK: [[C5:%[0-9]+]](s128) = G_IMPLICIT_DEF
; CHECK: [[C6:%[0-9]+]](s128) = G_INSERT [[C5]], [[VEC]](<2 x s32>), 0
; CHECK: [[C7:%[0-9]+]](s128) = G_INSERT [[C6]], [[C3]](s32), 64
; CHECK: [[C8:%[0-9]+]](s128) = G_INSERT [[C7]], [[C4]](s32), 96
; CHECK: [[EXT:%[0-9]+]](<2 x s32>) = G_EXTRACT [[C8]](s128), 0
; CHECK: G_EXTRACT_VECTOR_ELT [[EXT]](<2 x s32>)
  %vec = extractvalue %struct.v2s32.s32.s32 {<2 x i32><i32 1, i32 2>, i32 3, i32 4}, 0
  %elt = extractelement <2 x i32> %vec, i32 0
  ret i32 %elt
}
