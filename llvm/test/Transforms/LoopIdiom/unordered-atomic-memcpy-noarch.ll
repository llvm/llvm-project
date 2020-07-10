; RUN: opt -basic-aa -loop-idiom < %s -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

;; memcpy.atomic formation (atomic load & store) -- element size 2
;;  Will not create call due to a max element size of 0
define void @test1(i64 %Size) nounwind ssp {
; CHECK-LABEL: @test1(
; CHECK-NOT: call void @llvm.memcpy.element.unordered.atomic
; CHECK: store
; CHECK: ret void
bb.nph:
  %Base = alloca i16, i32 10000
  %Dest = alloca i16, i32 10000
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i16, i16* %Base, i64 %indvar
  %DestI = getelementptr i16, i16* %Dest, i64 %indvar
  %V = load atomic i16, i16* %I.0.014 unordered, align 2
  store atomic i16 %V, i16* %DestI unordered, align 2
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}
