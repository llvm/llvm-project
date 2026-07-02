; RUN: opt < %s -passes='expand-reductions,x86-partial-reduction' -mtriple=x86_64-unknown-unknown -mattr=+sse2 -S | FileCheck %s

; Isolate X86PartialReduction::tryByteSumReplacement on a positive shape.

@a = global [1024 x i8] zeroinitializer, align 16

; CHECK: call <2 x i64> @llvm.x86.sse2.psad.bw(
define i32 @byte_sum_v16_i32() nounwind {
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.phi = phi <16 x i32> [ zeroinitializer, %entry ], [ %add, %vector.body ]
  %p = getelementptr inbounds [1024 x i8], ptr @a, i64 0, i64 %index
  %wide.load = load <16 x i8>, ptr %p, align 16
  %z = zext <16 x i8> %wide.load to <16 x i32>
  %add = add nsw <16 x i32> %z, %vec.phi
  %index.next = add i64 %index, 16
  %cmp = icmp eq i64 %index.next, 1024
  br i1 %cmp, label %middle.block, label %vector.body

middle.block:
  %ext = call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %add)
  ret i32 %ext
}
