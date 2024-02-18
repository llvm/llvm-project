; Test loop tuning.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 -disable-block-placement | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -disable-block-placement \
; RUN:  | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-Z13

; Test that strength reduction is applied to addresses with a scale factor,
; but that indexed addressing can still be used.
define void @f1(ptr %dest, i32 %a) {
; CHECK-LABEL: f1:
; CHECK-NOT: sllg
; CHECK: st %r3, 400({{%r[1-5],%r[1-5]}})
; CHECK: br %r14
entry:
  br label %loop

loop:
  %index = phi i64 [ 0, %entry ], [ %next, %loop ]
  %ptr = getelementptr i32, ptr %dest, i64 %index
  store i32 %a, ptr %ptr
  %next = add i64 %index, 1
  %cmp = icmp ne i64 %next, 100
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; Test a loop that should be converted into dbr form and then use BRCT.
define void @f2(ptr %src, ptr %dest) {
; CHECK-LABEL: f2:
; CHECK: lhi [[REG:%r[0-5]]], 100
; CHECK: [[LABEL:\.[^:]*]]:{{.*}} %loop
; CHECK: brct [[REG]], [[LABEL]]
; CHECK: br %r14
entry:
  br label %loop

loop:
  %count = phi i32 [ 0, %entry ], [ %next, %loop.next ]
  %next = add i32 %count, 1
  %val = load volatile i32, ptr %src
  %cmp = icmp eq i32 %val, 0
  br i1 %cmp, label %loop.next, label %loop.store

loop.store:
  %add = add i32 %val, 1
  store volatile i32 %add, ptr %dest
  br label %loop.next

loop.next:
  %cont = icmp ne i32 %next, 100
  br i1 %cont, label %loop, label %exit

exit:
  ret void
}

; Like f2, but for BRCTG.
define void @f3(ptr %src, ptr %dest) {
; CHECK-LABEL: f3:
; CHECK: lghi [[REG:%r[0-5]]], 100
; CHECK: [[LABEL:\.[^:]*]]:{{.*}} %loop
; CHECK: brctg [[REG]], [[LABEL]]
; CHECK: br %r14
entry:
  br label %loop

loop:
  %count = phi i64 [ 0, %entry ], [ %next, %loop.next ]
  %next = add i64 %count, 1
  %val = load volatile i64, ptr %src
  %cmp = icmp eq i64 %val, 0
  br i1 %cmp, label %loop.next, label %loop.store

loop.store:
  %add = add i64 %val, 1
  store volatile i64 %add, ptr %dest
  br label %loop.next

loop.next:
  %cont = icmp ne i64 %next, 100
  br i1 %cont, label %loop, label %exit

exit:
  ret void
}

; Test a loop with a 64-bit decremented counter in which the 32-bit
; low part of the counter is used after the decrement.  This is an example
; of a subregister use being the only thing that blocks a conversion to BRCTG.
define void @f4(ptr %src, ptr %dest, ptr %dest2, i64 %count) {
; CHECK-LABEL: f4:
; CHECK: aghi [[REG:%r[0-5]]], -1
; CHECK: lr [[REG2:%r[0-5]]], [[REG]]
; CHECK: stg [[REG2]],
; CHECK: jne {{\..*}}
; CHECK: br %r14
entry:
  br label %loop

loop:
  %left = phi i64 [ %count, %entry ], [ %next, %loop.next ]
  store volatile i64 %left, ptr %dest2
  %val = load volatile i32, ptr %src
  %cmp = icmp eq i32 %val, 0
  br i1 %cmp, label %loop.next, label %loop.store

loop.store:
  %add = add i32 %val, 1
  store volatile i32 %add, ptr %dest
  br label %loop.next

loop.next:
  %next = add i64 %left, -1
  %ext = zext i32 %val to i64
  %shl = shl i64 %ext, 32
  %and = and i64 %next, 4294967295
  %or = or i64 %shl, %and
  store volatile i64 %or, ptr %dest2
  %cont = icmp ne i64 %next, 0
  br i1 %cont, label %loop, label %exit

exit:
  ret void
}

; Test that negative offsets are avoided for loads of floating point.
%s.float = type { float, float, float }
define void @f5(ptr nocapture %a,
                ptr nocapture readonly %b,
                i32 zeroext %S) {
; CHECK-Z13-LABEL: f5:
; CHECK-Z13-NOT: -{{[0-9]+}}(%r

entry:
  %cmp9 = icmp eq i32 %S, 0
  br i1 %cmp9, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                 ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:          ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                   ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                           ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %a1 = getelementptr inbounds %s.float, ptr %b, i64 %indvars.iv, i32 0
  %tmp = load float, ptr %a1, align 4
  %b4 = getelementptr inbounds %s.float, ptr %b, i64 %indvars.iv, i32 1
  %tmp1 = load float, ptr %b4, align 4
  %add = fadd float %tmp, %tmp1
  %c = getelementptr inbounds %s.float, ptr %b, i64 %indvars.iv, i32 2
  %tmp2 = load float, ptr %c, align 4
  %add7 = fadd float %add, %tmp2
  %a10 = getelementptr inbounds %s.float, ptr %a, i64 %indvars.iv, i32 0
  store float %add7, ptr %a10, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %S
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}

; Test that negative offsets are avoided for loads of double.
%s.double = type { double, double, double }
define void @f6(ptr nocapture %a,
                ptr nocapture readonly %b,
                i32 zeroext %S) {
; CHECK-Z13-LABEL: f6:
; CHECK-Z13-NOT: -{{[0-9]+}}(%r
entry:
  %cmp9 = icmp eq i32 %S, 0
  br i1 %cmp9, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                  ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:           ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                    ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                            ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %a1 = getelementptr inbounds %s.double, ptr %b, i64 %indvars.iv, i32 0
  %tmp = load double, ptr %a1, align 4
  %b4 = getelementptr inbounds %s.double, ptr %b, i64 %indvars.iv, i32 1
  %tmp1 = load double, ptr %b4, align 4
  %add = fadd double %tmp, %tmp1
  %c = getelementptr inbounds %s.double, ptr %b, i64 %indvars.iv, i32 2
  %tmp2 = load double, ptr %c, align 4
  %add7 = fadd double %add, %tmp2
  %a10 = getelementptr inbounds %s.double, ptr %a, i64 %indvars.iv, i32 0
  store double %add7, ptr %a10, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %S
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}

; Test that negative offsets are avoided for memory accesses of vector type.
%s.vec = type { <4 x i32>, <4 x i32>, <4 x i32> }
define void @f7(ptr nocapture %a,
                ptr nocapture readonly %b,
                i32 zeroext %S) {
; CHECK-Z13-LABEL: f7:
; CHECK-Z13-NOT: -{{[0-9]+}}(%r
entry:
  %cmp9 = icmp eq i32 %S, 0
  br i1 %cmp9, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                 ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:          ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                   ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                           ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %a1 = getelementptr inbounds %s.vec, ptr %b, i64 %indvars.iv, i32 0
  %tmp = load <4 x i32>, ptr %a1, align 4
  %b4 = getelementptr inbounds %s.vec, ptr %b, i64 %indvars.iv, i32 1
  %tmp1 = load <4 x i32>, ptr %b4, align 4
  %add = add <4 x i32> %tmp1, %tmp
  %c = getelementptr inbounds %s.vec, ptr %b, i64 %indvars.iv, i32 2
  %tmp2 = load <4 x i32>, ptr %c, align 4
  %add7 = add <4 x i32> %add, %tmp2
  %a10 = getelementptr inbounds %s.vec, ptr %a, i64 %indvars.iv, i32 0
  store <4 x i32> %add7, ptr %a10, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %S
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}

; Test that a memcpy loop does not get a lot of lays before each mvc (D12 and no index-reg).
%0 = type { %1, ptr }
%1 = type { ptr, ptr }
%2 = type <{ %3, i32, [4 x i8] }>
%3 = type { ptr, ptr, ptr }

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1) #0

define void @f8() {
; CHECK-Z13-LABEL: f8:
; CHECK-Z13: mvc
; CHECK-Z13-NEXT: mvc
; CHECK-Z13-NEXT: mvc
; CHECK-Z13-NEXT: mvc

bb:
  %tmp = load ptr, ptr undef, align 8
  br i1 undef, label %bb2, label %bb1

bb1:                                              ; preds = %bb
  br label %bb2

bb2:                                              ; preds = %bb1, %bb
  %tmp3 = phi ptr [ %tmp, %bb ], [ undef, %bb1 ]
  %tmp4 = phi ptr [ undef, %bb ], [ undef, %bb1 ]
  br label %bb5

bb5:                                              ; preds = %bb5, %bb2
  %tmp6 = phi ptr [ %tmp21, %bb5 ], [ %tmp3, %bb2 ]
  %tmp7 = phi ptr [ %tmp20, %bb5 ], [ %tmp4, %bb2 ]
  %tmp8 = getelementptr inbounds %0, ptr %tmp7, i64 -1
  %tmp9 = getelementptr inbounds %0, ptr %tmp6, i64 -1
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 8 %tmp9, ptr align 8 %tmp8, i64 24, i1 false)
  %tmp12 = getelementptr inbounds %0, ptr %tmp7, i64 -2
  %tmp13 = getelementptr inbounds %0, ptr %tmp6, i64 -2
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 8 %tmp13, ptr align 8 %tmp12, i64 24, i1 false)
  %tmp16 = getelementptr inbounds %0, ptr %tmp7, i64 -3
  %tmp17 = getelementptr inbounds %0, ptr %tmp6, i64 -3
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 8 %tmp17, ptr align 8 %tmp16, i64 24, i1 false)
  %tmp20 = getelementptr inbounds %0, ptr %tmp7, i64 -4
  %tmp21 = getelementptr inbounds %0, ptr %tmp6, i64 -4
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 8 %tmp21, ptr align 8 %tmp20, i64 24, i1 false)
  br label %bb5
}

; Test that a chsi does not need an aghik inside the loop (no index reg)
define void @f9() {
; CHECK-Z13-LABEL: f9:
; CHECK-Z13: # =>This Inner Loop Header: Depth=1
; CHECK-Z13-NOT: aghik
; CHECK-Z13: chsi

entry:
  br label %for.body.i63

for.body.i63:                                     ; preds = %for.inc.i, %entry
  %indvars.iv155.i = phi i64 [ 0, %entry ], [ %indvars.iv.next156.i.3, %for.inc.i ]
  %arrayidx.i62 = getelementptr inbounds i32, ptr undef, i64 %indvars.iv155.i
  %tmp = load i32, ptr %arrayidx.i62, align 4
  %cmp9.i = icmp eq i32 %tmp, 0
  br i1 %cmp9.i, label %for.inc.i, label %if.then10.i

if.then10.i:                                      ; preds = %for.body.i63
  unreachable

for.inc.i:                                        ; preds = %for.body.i63
  %indvars.iv.next156.i = or i64 %indvars.iv155.i, 1
  %arrayidx.i62.1 = getelementptr inbounds i32, ptr undef, i64 %indvars.iv.next156.i
  %tmp1 = load i32, ptr %arrayidx.i62.1, align 4
  %indvars.iv.next156.i.3 = add nsw i64 %indvars.iv155.i, 4
  br label %for.body.i63
}

; Test that offsets are in range for i128 memory accesses.
define void @fun10() {
; CHECK-Z13-LABEL: fun10:
; CHECK-Z13: # =>This Inner Loop Header: Depth=1
; CHECK-Z13-NOT: lay
entry:
  %A1 = alloca [3 x [7 x [10 x i128]]], align 8
  br label %for.body

for.body:                        ; preds = %for.body, %entry
  %IV = phi i64 [ 0, %entry ], [ %IV.next, %for.body ]
  %Addr1 = getelementptr inbounds [3 x [7 x [10 x i128]]], ptr %A1, i64 0, i64 %IV, i64 6, i64 6
  store i128 17174966165894859678, ptr %Addr1, align 8
  %Addr2 = getelementptr inbounds [3 x [7 x [10 x i128]]], ptr %A1, i64 0, i64 %IV, i64 6, i64 8
  store i128 17174966165894859678, ptr %Addr2, align 8
  %IV.next = add nuw nsw i64 %IV, 1
  %exitcond.not.i.i = icmp eq i64 %IV.next, 3
  br i1 %exitcond.not.i.i, label %exit, label %for.body

exit:                        ; preds = %for.body
  unreachable
}
