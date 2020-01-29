; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define signext i8 @func1(i8 signext %a, i8 signext %b) {
; CHECK-LABEL: func1:
; CHECK:       .LBB{{[0-9]+}}_5:
; CHECK-NEXT:    brle.w %s0, %s1, .LBB0_1
; CHECK-NEXT:  # %bb.2:
; CHECK-NEXT:    lea %s0, ret@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, ret@hi(%s0)
; CHECK-NEXT:    or %s0, 2, (0)1
; CHECK-NEXT:    bsic %lr, (,%s12)
; CHECK-NEXT:    br.l .LBB0_3
; CHECK:       .LBB{{[0-9]+}}_1:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK:       .LBB{{[0-9]+}}_3:
; CHECK-NEXT:    sla.w.sx %s0, %s0, 24
; CHECK-NEXT:    sra.w.sx %s0, %s0, 24
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %cmp = icmp sgt i8 %a, %b
  br i1 %cmp, label %on.true, label %join

on.true:
  %ret.val = tail call i32 @ret(i32 2)
  %r8 = trunc i32 %ret.val to i8
  br label %join

join:
  %r = phi i8 [ %r8, %on.true ], [ 0, %entry ]
  ret i8 %r
}

declare i32 @ret(i32)

define i32 @func2(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: func2:
; CHECK:       .LBB{{[0-9]+}}_5:
; CHECK-NEXT:    brle.w %s0, %s1, .LBB1_1
; CHECK-NEXT:  # %bb.2:
; CHECK-NEXT:    lea %s0, ret@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, ret@hi(%s0)
; CHECK-NEXT:    or %s0, 2, (0)1
; CHECK-NEXT:    bsic %lr, (,%s12)
; CHECK-NEXT:    br.l .LBB1_3
; CHECK:       .LBB{{[0-9]+}}_1:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK:       .LBB{{[0-9]+}}_3:
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %cmp = icmp sgt i16 %a, %b
  br i1 %cmp, label %on.true, label %join

on.true:
  %ret.val = tail call i32 @ret(i32 2)
  br label %join

join:
  %r = phi i32 [ %ret.val, %on.true ], [ 0, %entry ]
  ret i32 %r
}

define i32 @func3(i32 %a, i32 %b) {
; CHECK-LABEL: func3:
; CHECK:       .LBB{{[0-9]+}}_5:
; CHECK-NEXT:    brle.w %s0, %s1, .LBB2_1
; CHECK-NEXT:  # %bb.2:
; CHECK-NEXT:    lea %s0, ret@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, ret@hi(%s0)
; CHECK-NEXT:    or %s0, 2, (0)1
; CHECK-NEXT:    bsic %lr, (,%s12)
; CHECK-NEXT:    br.l .LBB2_3
; CHECK:       .LBB{{[0-9]+}}_1:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK:       .LBB{{[0-9]+}}_3:
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %on.true, label %join

on.true:
  %ret.val = tail call i32 @ret(i32 2)
  br label %join

join:
  %r = phi i32 [ %ret.val, %on.true ], [ 0, %entry ]
  ret i32 %r
}

define i32 @func4(i64 %a, i64 %b) {
; CHECK-LABEL: func4:
; CHECK:       .LBB{{[0-9]+}}_5:
; CHECK-NEXT:    brle.l %s0, %s1, .LBB3_1
; CHECK-NEXT:  # %bb.2:
; CHECK-NEXT:    lea %s0, ret@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, ret@hi(%s0)
; CHECK-NEXT:    or %s0, 2, (0)1
; CHECK-NEXT:    bsic %lr, (,%s12)
; CHECK-NEXT:    br.l .LBB3_3
; CHECK:       .LBB{{[0-9]+}}_1:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK:       .LBB{{[0-9]+}}_3:
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %cmp = icmp sgt i64 %a, %b
  br i1 %cmp, label %on.true, label %join

on.true:
  %ret.val = tail call i32 @ret(i32 2)
  br label %join

join:
  %r = phi i32 [ %ret.val, %on.true ], [ 0, %entry ]
  ret i32 %r
}

define i32 @func5(i8 zeroext %a, i8 zeroext %b) {
; CHECK-LABEL: func5:
; CHECK:       .LBB{{[0-9]+}}_5:
; CHECK-NEXT:    cmpu.w %s0, %s1, %s0
; CHECK-NEXT:    brle.w 0, %s0, .LBB4_1
; CHECK-NEXT:  # %bb.2:
; CHECK-NEXT:    lea %s0, ret@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, ret@hi(%s0)
; CHECK-NEXT:    or %s0, 2, (0)1
; CHECK-NEXT:    bsic %lr, (,%s12)
; CHECK-NEXT:    br.l .LBB4_3
; CHECK:       .LBB{{[0-9]+}}_1:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK:       .LBB{{[0-9]+}}_3:
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %cmp = icmp ugt i8 %a, %b
  br i1 %cmp, label %on.true, label %join

on.true:
  %ret.val = tail call i32 @ret(i32 2)
  br label %join

join:
  %r = phi i32 [ %ret.val, %on.true ], [ 0, %entry ]
  ret i32 %r
}

define i32 @func6(i16 zeroext %a, i16 zeroext %b) {
; CHECK-LABEL: func6:
; CHECK:       .LBB{{[0-9]+}}_5:
; CHECK-NEXT:    cmpu.w %s0, %s1, %s0
; CHECK-NEXT:    brle.w 0, %s0, .LBB5_1
; CHECK-NEXT:  # %bb.2:
; CHECK-NEXT:    lea %s0, ret@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, ret@hi(%s0)
; CHECK-NEXT:    or %s0, 2, (0)1
; CHECK-NEXT:    bsic %lr, (,%s12)
; CHECK-NEXT:    br.l .LBB5_3
; CHECK:       .LBB{{[0-9]+}}_1:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK:       .LBB{{[0-9]+}}_3:
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %cmp = icmp ugt i16 %a, %b
  br i1 %cmp, label %on.true, label %join

on.true:
  %ret.val = tail call i32 @ret(i32 2)
  br label %join

join:
  %r = phi i32 [ %ret.val, %on.true ], [ 0, %entry ]
  ret i32 %r
}

define i32 @func7(i32 %a, i32 %b) {
; CHECK-LABEL: func7:
; CHECK:       .LBB{{[0-9]+}}_5:
; CHECK-NEXT:    cmpu.w %s0, %s1, %s0
; CHECK-NEXT:    brle.w 0, %s0, .LBB6_1
; CHECK-NEXT:  # %bb.2:
; CHECK-NEXT:    lea %s0, ret@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, ret@hi(%s0)
; CHECK-NEXT:    or %s0, 2, (0)1
; CHECK-NEXT:    bsic %lr, (,%s12)
; CHECK-NEXT:    br.l .LBB6_3
; CHECK:       .LBB{{[0-9]+}}_1:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK:       .LBB{{[0-9]+}}_3:
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %cmp = icmp ugt i32 %a, %b
  br i1 %cmp, label %on.true, label %join

on.true:
  %ret.val = tail call i32 @ret(i32 2)
  br label %join

join:
  %r = phi i32 [ %ret.val, %on.true ], [ 0, %entry ]
  ret i32 %r
}

define i32 @func8(float %a, float %b) {
; CHECK-LABEL: func8:
; CHECK:       .LBB{{[0-9]+}}_5:
; CHECK-NEXT:    brlenan.s %s0, %s1, .LBB7_1
; CHECK-NEXT:  # %bb.2:
; CHECK-NEXT:    lea %s0, ret@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, ret@hi(%s0)
; CHECK-NEXT:    or %s0, 2, (0)1
; CHECK-NEXT:    bsic %lr, (,%s12)
; CHECK-NEXT:    br.l .LBB7_3
; CHECK:       .LBB{{[0-9]+}}_1:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK:       .LBB{{[0-9]+}}_3:
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %cmp = fcmp ogt float %a, %b
  br i1 %cmp, label %on.true, label %join

on.true:
  %ret.val = tail call i32 @ret(i32 2)
  br label %join

join:
  %r = phi i32 [ %ret.val, %on.true ], [ 0, %entry ]
  ret i32 %r
}

define i32 @func9(double %a, double %b) {
; CHECK-LABEL: func9:
; CHECK:       .LBB{{[0-9]+}}_5:
; CHECK-NEXT:    brlenan.d %s0, %s1, .LBB8_1
; CHECK-NEXT:  # %bb.2:
; CHECK-NEXT:    lea %s0, ret@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, ret@hi(%s0)
; CHECK-NEXT:    or %s0, 2, (0)1
; CHECK-NEXT:    bsic %lr, (,%s12)
; CHECK-NEXT:    br.l .LBB8_3
; CHECK:       .LBB{{[0-9]+}}_1:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK:       .LBB{{[0-9]+}}_3:
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %cmp = fcmp ogt double %a, %b
  br i1 %cmp, label %on.true, label %join

on.true:
  %ret.val = tail call i32 @ret(i32 2)
  br label %join

join:
  %r = phi i32 [ %ret.val, %on.true ], [ 0, %entry ]
  ret i32 %r
}

define i32 @func10(double %a, double %b) {
; CHECK-LABEL: func10:
; CHECK:       .LBB{{[0-9]+}}_5:
; CHECK-NEXT:    lea.sl %s1, 1075052544
; CHECK-NEXT:    brlenan.d %s0, %s1, .LBB9_1
; CHECK-NEXT:  # %bb.2:
; CHECK-NEXT:    lea %s0, ret@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, ret@hi(%s0)
; CHECK-NEXT:    or %s0, 2, (0)1
; CHECK-NEXT:    bsic %lr, (,%s12)
; CHECK-NEXT:    br.l .LBB9_3
; CHECK:       .LBB{{[0-9]+}}_1:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK:       .LBB{{[0-9]+}}_3:
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %cmp = fcmp ogt double %a, 5.000000e+00
  br i1 %cmp, label %on.true, label %join

on.true:
  %ret.val = tail call i32 @ret(i32 2)
  br label %join

join:
  %r = phi i32 [ %ret.val, %on.true ], [ 0, %entry ]
  ret i32 %r
}
