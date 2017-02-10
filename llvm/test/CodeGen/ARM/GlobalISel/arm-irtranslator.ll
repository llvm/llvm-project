; RUN: llc -mtriple arm-unknown -mattr=+vfp2 -global-isel -stop-after=irtranslator %s -o - | FileCheck %s

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

define i32 @test_stack_args(i32 %p0, i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5) {
; CHECK-LABEL: name: test_stack_args
; CHECK: fixedStack:
; CHECK-DAG: id: [[P4:[0-9]]]{{.*}}offset: 0{{.*}}size: 4
; CHECK-DAG: id: [[P5:[0-9]]]{{.*}}offset: 4{{.*}}size: 4
; CHECK: liveins: %r0, %r1, %r2, %r3
; CHECK: [[VREGP2:%[0-9]+]]{{.*}} = COPY %r2
; CHECK: [[FIP5:%[0-9]+]]{{.*}} = G_FRAME_INDEX %fixed-stack.[[P5]]
; CHECK: [[VREGP5:%[0-9]+]]{{.*}} = G_LOAD [[FIP5]]
; CHECK: [[SUM:%[0-9]+]]{{.*}} = G_ADD [[VREGP2]], [[VREGP5]]
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
; CHECK: [[VREGP1:%[0-9]+]]{{.*}} = COPY %r1
; CHECK: [[FIP5:%[0-9]+]]{{.*}} = G_FRAME_INDEX %fixed-stack.[[P5]]
; CHECK: [[VREGP5:%[0-9]+]]{{.*}} = G_LOAD [[FIP5]](p0)
; CHECK: [[SUM:%[0-9]+]]{{.*}} = G_ADD [[VREGP1]], [[VREGP5]]
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
; CHECK: [[VREGP2:%[0-9]+]]{{.*}} = COPY %r2
; CHECK: [[FIP4:%[0-9]+]]{{.*}} = G_FRAME_INDEX %fixed-stack.[[P4]]
; CHECK: [[VREGP4:%[0-9]+]]{{.*}} = G_LOAD [[FIP4]](p0)
; CHECK: [[SUM:%[0-9]+]]{{.*}} = G_ADD [[VREGP2]], [[VREGP4]]
; CHECK: %r0 = COPY [[SUM]]
; CHECK: BX_RET 14, _, implicit %r0
entry:
  %sum = add i8 %p2, %p4
  ret i8 %sum
}

define i16 @test_ptr_arg(i16* %p) {
; CHECK-LABEL: name: test_ptr_arg
; CHECK: liveins: %r0
; CHECK: [[VREGP:%[0-9]+]](p0) = COPY %r0
; CHECK: [[VREGV:%[0-9]+]](s16) = G_LOAD [[VREGP]](p0)
entry:
  %v = load i16, i16* %p
  ret i16 %v
}

define i32* @test_ptr_ret(i32** %p) {
; Test pointer returns and pointer-to-pointer arguments
; CHECK-LABEL: name: test_ptr_ret
; CHECK: liveins: %r0
; CHECK: [[VREGP:%[0-9]+]](p0) = COPY %r0
; CHECK: [[VREGV:%[0-9]+]](p0) = G_LOAD [[VREGP]](p0)
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
; CHECK: [[FIP:%[0-9]+]]{{.*}} = G_FRAME_INDEX %fixed-stack.[[P]]
; CHECK: [[VREGP:%[0-9]+]](p0) = G_LOAD [[FIP]](p0)
; CHECK: [[VREGV:%[0-9]+]](s32) = G_LOAD [[VREGP]](p0)
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
; CHECK: [[VREGP5:%[0-9]+]](s32) = G_LOAD [[FIP5]](p0)
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
; CHECK: [[VREGQ1:%[0-9]+]](s32) = G_LOAD [[FIQ1]](p0)
; CHECK: [[VREGV:%[0-9]+]](s32) = G_FADD [[VREGP1]], [[VREGQ1]]
; CHECK: %s0 = COPY [[VREGV]]
; CHECK: BX_RET 14, _, implicit %s0
entry:
  %v = fadd float %p1, %q1
  ret float %v
}

