; RUN: opt -S -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -enable-interleaved-mem-accesses=true < %s

; Make sure the vectorizer can handle this loop: The strided load is only used
; by the loop's exit condition, which is not vectorized, and is therefore
; considered uniform while also forming an interleave group.
  
%0 = type { i32 ()*, i32 }

@0 = internal unnamed_addr constant [59 x %0] [%0 zeroinitializer,
%0 zeroinitializer, %0 zeroinitializer, %0 zeroinitializer,
%0 zeroinitializer, %0 zeroinitializer, %0 zeroinitializer,
%0 zeroinitializer, %0 zeroinitializer, %0 zeroinitializer,
%0 zeroinitializer, %0 zeroinitializer, %0 zeroinitializer,
%0 zeroinitializer, %0 zeroinitializer, %0 zeroinitializer,
%0 zeroinitializer, %0 zeroinitializer, %0 zeroinitializer,
%0 zeroinitializer, %0 zeroinitializer, %0 zeroinitializer,
%0 zeroinitializer, %0 zeroinitializer, %0 zeroinitializer,
%0 zeroinitializer, %0 zeroinitializer, %0 zeroinitializer,
%0 zeroinitializer, %0 zeroinitializer, %0 zeroinitializer,
%0 zeroinitializer, %0 zeroinitializer, %0 zeroinitializer,
%0 zeroinitializer, %0 zeroinitializer, %0 zeroinitializer,
%0 zeroinitializer, %0 zeroinitializer, %0 zeroinitializer,
%0 zeroinitializer, %0 zeroinitializer, %0 zeroinitializer,
%0 zeroinitializer, %0 zeroinitializer, %0 zeroinitializer,
%0 zeroinitializer, %0 zeroinitializer, %0 zeroinitializer,
%0 zeroinitializer, %0 zeroinitializer, %0 zeroinitializer,
%0 {i32 ()* null, i32 258}, %0 zeroinitializer, %0 zeroinitializer,
%0 zeroinitializer, %0 zeroinitializer, %0 zeroinitializer,
%0 zeroinitializer], align 8

define dso_local void @test_dead_load(i32 %arg) {
; CHECK-LABEL: @test_dead_load(
; CHECK: vector.body:
; CHECK: %wide.vec = load <16 x i32>, <16 x i32>* %3, align 8
; CHECK: %strided.vec = shufflevector <16 x i32> %wide.vec, <16 x i32> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
bb1:
  br label %bb2

bb2:
  %tmp = phi %0* [ %tmp6, %bb2 ], [ getelementptr inbounds ([59 x %0], [59 x %0]* @0, i64 0, i64 0), %bb1 ]
  %tmp3 = getelementptr inbounds %0, %0* %tmp, i64 0, i32 1
  %tmp4 = load i32, i32* %tmp3, align 8
  %tmp5 = icmp eq i32 %tmp4, 258
  %tmp6 = getelementptr inbounds %0, %0* %tmp, i64 1
  br i1 %tmp5, label %bb65, label %bb2

bb65:
  unreachable
}
