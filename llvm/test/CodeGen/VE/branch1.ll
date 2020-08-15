; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define signext i8 @func1(i8 signext %a, i8 signext %b) {
; CHECK-LABEL: func1:
; CHECK:       .LBB{{[0-9]+}}_5:
; CHECK-NEXT:    brle.w %s0, %s1, .LBB{{[0-9]+}}_1
; CHECK-NEXT:  # %bb.2: # %on.true
; CHECK-NEXT:    lea %s0, ret@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, ret@hi(, %s0)
; CHECK-NEXT:    or %s0, 2, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    br.l.t .LBB{{[0-9]+}}_3
; CHECK-NEXT:  .LBB{{[0-9]+}}_1:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:  .LBB{{[0-9]+}}_3: # %join
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
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
; CHECK-NEXT:    brle.w %s0, %s1, .LBB{{[0-9]+}}_1
; CHECK-NEXT:  # %bb.2: # %on.true
; CHECK-NEXT:    lea %s0, ret@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, ret@hi(, %s0)
; CHECK-NEXT:    or %s0, 2, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    br.l.t .LBB{{[0-9]+}}_3
; CHECK-NEXT:  .LBB{{[0-9]+}}_1:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:  .LBB{{[0-9]+}}_3: # %join
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
; CHECK-NEXT:    brle.w %s0, %s1, .LBB{{[0-9]+}}_1
; CHECK-NEXT:  # %bb.2: # %on.true
; CHECK-NEXT:    lea %s0, ret@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, ret@hi(, %s0)
; CHECK-NEXT:    or %s0, 2, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    br.l.t .LBB{{[0-9]+}}_3
; CHECK-NEXT:  .LBB{{[0-9]+}}_1:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:  .LBB{{[0-9]+}}_3: # %join
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
; CHECK-NEXT:    brle.l %s0, %s1, .LBB{{[0-9]+}}_1
; CHECK-NEXT:  # %bb.2: # %on.true
; CHECK-NEXT:    lea %s0, ret@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, ret@hi(, %s0)
; CHECK-NEXT:    or %s0, 2, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    br.l.t .LBB{{[0-9]+}}_3
; CHECK-NEXT:  .LBB{{[0-9]+}}_1:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:  .LBB{{[0-9]+}}_3: # %join
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
; CHECK-NEXT:    brle.w 0, %s0, .LBB{{[0-9]+}}_1
; CHECK-NEXT:  # %bb.2: # %on.true
; CHECK-NEXT:    lea %s0, ret@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, ret@hi(, %s0)
; CHECK-NEXT:    or %s0, 2, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    br.l.t .LBB{{[0-9]+}}_3
; CHECK-NEXT:  .LBB{{[0-9]+}}_1:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:  .LBB{{[0-9]+}}_3: # %join
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
; CHECK-NEXT:    brle.w 0, %s0, .LBB{{[0-9]+}}_1
; CHECK-NEXT:  # %bb.2: # %on.true
; CHECK-NEXT:    lea %s0, ret@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, ret@hi(, %s0)
; CHECK-NEXT:    or %s0, 2, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    br.l.t .LBB{{[0-9]+}}_3
; CHECK-NEXT:  .LBB{{[0-9]+}}_1:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:  .LBB{{[0-9]+}}_3: # %join
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
; CHECK-NEXT:    brle.w 0, %s0, .LBB{{[0-9]+}}_1
; CHECK-NEXT:  # %bb.2: # %on.true
; CHECK-NEXT:    lea %s0, ret@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, ret@hi(, %s0)
; CHECK-NEXT:    or %s0, 2, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    br.l.t .LBB{{[0-9]+}}_3
; CHECK-NEXT:  .LBB{{[0-9]+}}_1:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:  .LBB{{[0-9]+}}_3: # %join
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
; CHECK-NEXT:    brlenan.s %s0, %s1, .LBB{{[0-9]+}}_1
; CHECK-NEXT:  # %bb.2: # %on.true
; CHECK-NEXT:    lea %s0, ret@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, ret@hi(, %s0)
; CHECK-NEXT:    or %s0, 2, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    br.l.t .LBB{{[0-9]+}}_3
; CHECK-NEXT:  .LBB{{[0-9]+}}_1:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:  .LBB{{[0-9]+}}_3: # %join
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
; CHECK-NEXT:    brlenan.d %s0, %s1, .LBB{{[0-9]+}}_1
; CHECK-NEXT:  # %bb.2: # %on.true
; CHECK-NEXT:    lea %s0, ret@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, ret@hi(, %s0)
; CHECK-NEXT:    or %s0, 2, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    br.l.t .LBB{{[0-9]+}}_3
; CHECK-NEXT:  .LBB{{[0-9]+}}_1:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:  .LBB{{[0-9]+}}_3: # %join
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
; CHECK-NEXT:    brlenan.d %s0, %s1, .LBB{{[0-9]+}}_1
; CHECK-NEXT:  # %bb.2: # %on.true
; CHECK-NEXT:    lea %s0, ret@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, ret@hi(, %s0)
; CHECK-NEXT:    or %s0, 2, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    br.l.t .LBB{{[0-9]+}}_3
; CHECK-NEXT:  .LBB{{[0-9]+}}_1:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:  .LBB{{[0-9]+}}_3: # %join
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
