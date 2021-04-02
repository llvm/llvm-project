; RUN: opt -S < %s -passes=licm | FileCheck %s

declare i8* @llvm.launder.invariant.group.p0i8(i8* %a)

; CHECK-LABEL: define{{.*}}@f
define void @f(i32* %x) {
; CHECK: entry:
; CHECK-NOT: {{.*}}:
; CHECK: load {{.*}} !invariant.group
entry:
  %x_i8 = bitcast i32* %x to i8*
  %x_i8_inv = call i8* @llvm.launder.invariant.group.p0i8(i8* %x_i8)
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.04 = phi i32 [ 0, %entry ], [ %inc, %for.body ]

  %x_inv = bitcast i8* %x_i8_inv to i32*
  %0 = load i32, i32* %x_inv, !invariant.group !0

  call void @a(i32 %0)
  %inc = add nuw nsw i32 %i.04, 1
  %exitcond.not = icmp eq i32 %inc, 100
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: define{{.*}}@g
define void @g(i32* %x) {
; CHECK: for.body:
; CHECK-NOT: {{.*}}:
; CHECK: load {{.*}} !invariant.group
entry:
  %x_i8 = bitcast i32* %x to i8*
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.04 = phi i32 [ 0, %entry ], [ %inc, %for.body ]

  %x_i8_inv = call i8* @llvm.launder.invariant.group.p0i8(i8* %x_i8)
  %x_inv = bitcast i8* %x_i8_inv to i32*

  %0 = load i32, i32* %x_inv, !invariant.group !0

  call void @a(i32 %0)
  %inc = add nuw nsw i32 %i.04, 1
  %exitcond.not = icmp eq i32 %inc, 100
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare void @a(i32)

!0 = !{}
