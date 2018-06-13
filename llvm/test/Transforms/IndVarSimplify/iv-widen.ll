; RUN: opt < %s -indvars -S | FileCheck %s
; RUN: opt -lcssa -loop-simplify -S < %s | opt -S -passes='require<targetir>,require<scalar-evolution>,require<domtree>,loop(indvars)'

; Provide legal integer types.
target datalayout = "n8:16:32:64"


target triple = "x86_64-apple-darwin"

declare void @use(i64 %x)

; CHECK-LABEL: @loop_0
; CHECK-LABEL: B18:
; Only one phi now.
; CHECK: phi i64
; CHECK-NOT: phi
; One trunc for the gep.
; CHECK: trunc i64 %indvars.iv to i32
; One trunc for the dummy() call.
; CHECK-LABEL: exit24:
; CHECK: trunc i64 {{.*}}lcssa.wide to i32
define void @loop_0(i32* %a) {
Prologue:
  br i1 undef, label %B18, label %B6

B18:                                        ; preds = %B24, %Prologue
  %.02 = phi i32 [ 0, %Prologue ], [ %tmp33, %B24 ]
  %tmp23 = zext i32 %.02 to i64
  call void @use(i64 %tmp23)
  %tmp33 = add i32 %.02, 1
  %o = getelementptr i32, i32* %a, i32 %.02
  %v = load i32, i32* %o
  %t = icmp eq i32 %v, 0
  br i1 %t, label %exit24, label %B24

B24:                                        ; preds = %B18
  %t2 = icmp eq i32 %tmp33, 20
  br i1 %t2, label %B6, label %B18

B6:                                       ; preds = %Prologue
  ret void

exit24:                      ; preds = %B18
  call void @dummy(i32 %.02)
  unreachable
}

; Make sure that dead zext is removed and no widening happens.
; CHECK-LABEL: @loop_0.dead
; CHECK: phi i32
; CHECK-NOT: zext
; CHECK-NOT: trunc
define void @loop_0.dead(i32* %a) {
Prologue:
  br i1 undef, label %B18, label %B6

B18:                                        ; preds = %B24, %Prologue
  %.02 = phi i32 [ 0, %Prologue ], [ %tmp33, %B24 ]
  %tmp23 = zext i32 %.02 to i64
  %tmp33 = add i32 %.02, 1
  %o = getelementptr i32, i32* %a, i32 %.02
  %v = load i32, i32* %o
  %t = icmp eq i32 %v, 0
  br i1 %t, label %exit24, label %B24

B24:                                        ; preds = %B18
  %t2 = icmp eq i32 %tmp33, 20
  br i1 %t2, label %B6, label %B18

B6:                                       ; preds = %Prologue
  ret void

exit24:                      ; preds = %B18
  call void @dummy(i32 %.02)
  unreachable
}

define void @loop_1(i32 %lim) {
; CHECK-LABEL: @loop_1(
 entry:
  %entry.cond = icmp ne i32 %lim, 0
  br i1 %entry.cond, label %loop, label %leave

 loop:
; CHECK: loop:
; CHECK:  %indvars.iv = phi i64 [ 1, %loop.preheader ], [ %indvars.iv.next, %loop ]
; CHECK:  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK:  [[IV_INC:%[^ ]+]] = add nsw i64 %indvars.iv, -1
; CHECK:  call void @dummy.i64(i64 [[IV_INC]])

  %iv = phi i32 [ 1, %entry ], [ %iv.inc, %loop ]
  %iv.inc = add i32 %iv, 1
  %iv.inc.sub = add i32 %iv, -1
  %iv.inc.sub.zext = zext i32 %iv.inc.sub to i64
  call void @dummy.i64(i64 %iv.inc.sub.zext)
  %be.cond = icmp ult i32 %iv.inc, %lim
  br i1 %be.cond, label %loop, label %leave

 leave:
  ret void
}

declare void @dummy(i32)
declare void @dummy.i64(i64)


define void @loop_2(i32 %size, i32 %nsteps, i32 %hsize, i32* %lined, i8 %tmp1) {
; CHECK-LABEL: @loop_2(
entry:
  %cmp215 = icmp sgt i32 %size, 1
  %tmp0 = bitcast i32* %lined to i8*
  br label %for.body

for.body:
  %j = phi i32 [ 0, %entry ], [ %inc6, %for.inc ]
  %mul = mul nsw i32 %j, %size
  %add = add nsw i32 %mul, %hsize
  br i1 %cmp215, label %for.body2, label %for.inc

; check that the induction variable of the inner loop has been widened after indvars.
; CHECK:  [[INNERLOOPINV:%[^ ]+]] = add nsw i64
; CHECK: for.body2:
; CHECK-NEXT:  %indvars.iv = phi i64 [ 1, %for.body2.preheader ], [ %indvars.iv.next, %for.body2 ]
; CHECK-NEXT:  [[WIDENED:%[^ ]+]] = add nsw i64 [[INNERLOOPINV]], %indvars.iv
; CHECK-NEXT:  %add.ptr = getelementptr inbounds i8, i8* %tmp0, i64 [[WIDENED]]
for.body2:
  %k = phi i32 [ %inc, %for.body2 ], [ 1, %for.body ]
  %add4 = add nsw i32 %add, %k
  %idx.ext = sext i32 %add4 to i64
  %add.ptr = getelementptr inbounds i8, i8* %tmp0, i64 %idx.ext
  store i8 %tmp1, i8* %add.ptr, align 1
  %inc = add nsw i32 %k, 1
  %cmp2 = icmp slt i32 %inc, %size
  br i1 %cmp2, label %for.body2, label %for.body3

; check that the induction variable of the inner loop has been widened after indvars.
; CHECK: for.body3.preheader:
; CHECK:  [[INNERLOOPINV:%[^ ]+]] = zext i32
; CHECK: for.body3:
; CHECK-NEXT:  %indvars.iv2 = phi i64 [ 1, %for.body3.preheader ], [ %indvars.iv.next3, %for.body3 ]
; CHECK-NEXT:  [[WIDENED:%[^ ]+]] = add nuw nsw i64 [[INNERLOOPINV]], %indvars.iv2
; CHECK-NEXT:  %add.ptr2 = getelementptr inbounds i8, i8* %tmp0, i64 [[WIDENED]]
for.body3:
  %l = phi i32 [ %inc2, %for.body3 ], [ 1, %for.body2 ]
  %add5 = add nuw i32 %add, %l
  %idx.ext2 = zext i32 %add5 to i64
  %add.ptr2 = getelementptr inbounds i8, i8* %tmp0, i64 %idx.ext2
  store i8 %tmp1, i8* %add.ptr2, align 1
  %inc2 = add nsw i32 %l, 1
  %cmp3 = icmp slt i32 %inc2, %size
  br i1 %cmp3, label %for.body3, label %for.inc

for.inc:
  %inc6 = add nsw i32 %j, 1
  %cmp = icmp slt i32 %inc6, %nsteps
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:
  ret void
}
