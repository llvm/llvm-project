; RUN: opt < %s -passes=bounds-checking -S | FileCheck %s
@array = internal global [1819 x i16] zeroinitializer, section ".bss,bss"
@offsets = external dso_local global [10 x i16]

; CHECK-LABEL: @test
define dso_local void @test() {
bb1:
  br label %bb19

bb20:
  %_tmp819 = load i16, ptr null
; CHECK: br {{.*}} %trap
  %_tmp820 = sub nsw i16 9, %_tmp819
  %_tmp821 = sext i16 %_tmp820 to i64
  %_tmp822 = getelementptr [10 x i16], ptr @offsets, i16 0, i64 %_tmp821
  %_tmp823 = load i16, ptr %_tmp822
  br label %bb33

bb34:
  %_tmp907 = zext i16 %i__7.107.0 to i64
  %_tmp908 = getelementptr [1819 x i16], ptr @array, i16 0, i64 %_tmp907
  store i16 0, ptr %_tmp908
; CHECK: br {{.*}} %trap
  %_tmp910 = add i16 %i__7.107.0, 1
  br label %bb33

bb33:
  %i__7.107.0 = phi i16 [ undef, %bb20 ], [ %_tmp910, %bb34 ]
  %_tmp913 = add i16 %_tmp823, 191
  %_tmp914 = icmp ult i16 %i__7.107.0, %_tmp913
  br i1 %_tmp914, label %bb34, label %bb19

bb19:
  %_tmp976 = icmp slt i16 0, 10
  br i1 %_tmp976, label %bb20, label %bb39

bb39:
  ret void
}

@e = dso_local local_unnamed_addr global [1 x i16] zeroinitializer, align 1

; CHECK-LABEL: @test2
define dso_local void @test2() local_unnamed_addr {
entry:
  br label %while.cond1.preheader

while.cond1.preheader:
  %0 = phi i16 [ undef, %entry ], [ %inc, %while.end ]
  %1 = load i16, ptr undef, align 1
; CHECK: br {{.*}} %trap
  br label %while.end

while.end:
  %inc = add nsw i16 %0, 1
  %arrayidx = getelementptr inbounds [1 x i16], ptr @e, i16 0, i16
 %0
  %2 = load i16, ptr %arrayidx, align 1
; CHECK: or i1
; CHECK-NEXT: br {{.*}} %trap
  br i1 false, label %while.end6, label %while.cond1.preheader

while.end6:
  ret void
}
