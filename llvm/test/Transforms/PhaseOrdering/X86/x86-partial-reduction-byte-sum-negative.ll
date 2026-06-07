; RUN: opt < %s -passes='expand-reductions,x86-partial-reduction' -mtriple=x86_64-unknown-unknown -mattr=+avx2 -S | FileCheck %s

; Shapes that tryByteSumReplacement must not rewrite.

@a = global [1024 x i8] zeroinitializer, align 16

; CHECK-LABEL: @byte_sum_v8_i32
; CHECK-NOT: psad.bw
define i32 @byte_sum_v8_i32() nounwind {
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.phi = phi <8 x i32> [ zeroinitializer, %entry ], [ %add, %vector.body ]
  %p = getelementptr inbounds [1024 x i8], ptr @a, i64 0, i64 %index
  %wide.load = load <8 x i8>, ptr %p, align 8
  %z = zext <8 x i8> %wide.load to <8 x i32>
  %add = add nsw <8 x i32> %z, %vec.phi
  %index.next = add i64 %index, 8
  %cmp = icmp eq i64 %index.next, 1024
  br i1 %cmp, label %middle.block, label %vector.body

middle.block:
  %ext = call i32 @llvm.vector.reduce.add.v8i32(<8 x i32> %add)
  ret i32 %ext
}

; CHECK-LABEL: @byte_sum_v24_i32
; CHECK-NOT: psad.bw
define i32 @byte_sum_v24_i32() nounwind {
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.phi = phi <24 x i32> [ zeroinitializer, %entry ], [ %add, %vector.body ]
  %p = getelementptr inbounds [1024 x i8], ptr @a, i64 0, i64 %index
  %wide.load = load <24 x i8>, ptr %p, align 8
  %z = zext <24 x i8> %wide.load to <24 x i32>
  %add = add nsw <24 x i32> %z, %vec.phi
  %index.next = add i64 %index, 24
  %cmp = icmp eq i64 %index.next, 1024
  br i1 %cmp, label %middle.block, label %vector.body

middle.block:
  %ext = call i32 @llvm.vector.reduce.add.v24i32(<24 x i32> %add)
  ret i32 %ext
}
