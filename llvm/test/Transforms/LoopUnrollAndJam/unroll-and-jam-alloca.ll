target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
; RUN: opt -passes=loop-unroll-and-jam -allow-unroll-and-jam -unroll-and-jam-count=4 -unroll-remainder < %s -S | FileCheck %s

; The alloca and lifetime use are problematic here, because the unroll
; duplication will require a merge. A lifetime intrinsic can't have a
; phi definition. The lifetime intrinsic can be removed.

; CHECK: [[xtraiter:%.+]] = and i32 {{.*}}, 3
; CHECK-NOT: call{{.*}}@llvm.lifetime

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #0

define void @test1(i32 %I) {
entry:
  br label %for.outer

for.outer:                                        ; preds = %for.latch, %entry
  %i = phi i32 [ %add8, %for.latch ], [ 0, %entry ]
  br label %for.inner

for.inner:                                        ; preds = %for.inner, %for.outer
  %alloca = alloca i32, align 4
  br i1 true, label %for.latch, label %for.inner

for.latch:                                        ; preds = %for.inner
  %add8 = add i32 %i, 1
  %exitcond25 = icmp eq i32 %i, %I
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer

for.end.loopexit:                                 ; preds = %for.latch
  call void @llvm.lifetime.start.p0(ptr %alloca)
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
