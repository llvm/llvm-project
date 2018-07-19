; RUN: llc < %s -o - -mcpu=generic -mtriple=x86_64-apple-darwin -mattr=+sse2 | FileCheck %s
; RUN: llc < %s -o - -mcpu=generic -mtriple=x86_64-apple-darwin -mattr=+sse4.2 | FileCheck %s

; For a setult against a constant, turn it into a setule and lower via psubusw.

define void @loop_no_const_reload(<2 x i64>*  %in, <2 x i64>* %out, i32 %n) {
; CHECK: .short 25
; CHECK-NEXT: .short 25
; CHECK-NEXT: .short 25
; CHECK-NEXT: .short 25
; CHECK-NEXT: .short 25
; CHECK-NEXT: .short 25
; CHECK-NEXT: .short 25
; CHECK-NEXT: .short 25
; CHECK-LABEL: loop_no_const_reload:
; CHECK: psubusw

; Constant is no longer clobbered so no need to reload it in the loop.

; CHECK-NOT: movdqa {{%xmm[0-9]+}}, {{%xmm[0-9]+}}

entry:
  %cmp9 = icmp eq i32 %n, 0
  br i1 %cmp9, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx1 = getelementptr inbounds <2 x i64>, <2 x i64>* %in, i64 %indvars.iv
  %arrayidx1.val = load <2 x i64>, <2 x i64>* %arrayidx1, align 16
  %0 = bitcast <2 x i64> %arrayidx1.val to <8 x i16>
  %cmp.i.i = icmp ult <8 x i16> %0, <i16 26, i16 26, i16 26, i16 26, i16 26, i16 26, i16 26, i16 26>
  %sext.i.i = sext <8 x i1> %cmp.i.i to <8 x i16>
  %1 = bitcast <8 x i16> %sext.i.i to <2 x i64>
  %arrayidx5 = getelementptr inbounds <2 x i64>, <2 x i64>* %out, i64 %indvars.iv
  store <2 x i64> %1, <2 x i64>* %arrayidx5, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; Be careful if decrementing the constant would undeflow.

define void @loop_const_folding_underflow(<2 x i64>*  %in, <2 x i64>* %out, i32 %n) {
; CHECK-NOT: .short 25
; CHECK-LABEL: loop_const_folding_underflow:
; CHECK-NOT: psubusw
entry:
  %cmp9 = icmp eq i32 %n, 0
  br i1 %cmp9, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx1 = getelementptr inbounds <2 x i64>, <2 x i64>* %in, i64 %indvars.iv
  %arrayidx1.val = load <2 x i64>, <2 x i64>* %arrayidx1, align 16
  %0 = bitcast <2 x i64> %arrayidx1.val to <8 x i16>
  %cmp.i.i = icmp ult <8 x i16> %0, <i16 0, i16 26, i16 26, i16 26, i16 26, i16 26, i16 26, i16 26>
  %sext.i.i = sext <8 x i1> %cmp.i.i to <8 x i16>
  %1 = bitcast <8 x i16> %sext.i.i to <2 x i64>
  %arrayidx5 = getelementptr inbounds <2 x i64>, <2 x i64>* %out, i64 %indvars.iv
  store <2 x i64> %1, <2 x i64>* %arrayidx5, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; Test for PSUBUSB

define <16 x i8> @test_ult_byte(<16 x i8> %a) {
; CHECK: .space 16,10
; CHECK-LABEL: test_ult_byte:
; CHECK: psubus
entry:
  %icmp = icmp ult <16 x i8> %a, <i8 11, i8 11, i8 11, i8 11, i8 11, i8 11, i8 11, i8 11, i8 11, i8 11, i8 11, i8 11, i8 11, i8 11, i8 11, i8 11>
  %sext = sext <16 x i1> %icmp to <16 x i8>
  ret <16 x i8> %sext
}

; Only do this when we can turn the comparison into a setule.  I.e. not for
; register operands.

define <8 x i16> @test_ult_register(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_ult_register:
; CHECK-NOT: psubus
entry:
  %icmp = icmp ult <8 x i16> %a, %b
  %sext = sext <8 x i1> %icmp to <8 x i16>
  ret <8 x i16> %sext
}
