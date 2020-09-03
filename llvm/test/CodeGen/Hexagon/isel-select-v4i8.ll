; RUN: llc -march=hexagon < %s | FileCheck %s

; This used to fail:
; LLVM ERROR: Cannot select: t54: v4i8 = select t50, t53, t52

; CHECK: jumpr r31

target triple = "hexagon"

@g0 = external dso_local unnamed_addr constant [41 x i8], align 1
define dso_local void @f0() local_unnamed_addr #0 {
b0:
  %v0 = load <16 x i32>, <16 x i32>* undef, align 16
  %v1 = icmp eq <16 x i32> %v0, zeroinitializer
  %v2 = or <16 x i1> %v1, zeroinitializer
  %v3 = or <16 x i1> %v2, zeroinitializer
  %v4 = or <16 x i1> %v3, zeroinitializer
  %v5 = shufflevector <16 x i1> %v4, <16 x i1> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %v6 = or <16 x i1> %v4, %v5
  %v7 = extractelement <16 x i1> %v6, i32 0
  %v8 = or i1 %v7, undef
  %v9 = or i1 %v8, undef
  br i1 %v9, label %b2, label %b1

b1:                                               ; preds = %b0
  call void (i8*, ...) @f1(i8* getelementptr inbounds ([41 x i8], [41 x i8]* @g0, i32 0, i32 0))
  unreachable

b2:                                               ; preds = %b0
  ret void
}
declare dso_local void @f1(i8*, ...) local_unnamed_addr #1

attributes #0 = { "target-cpu"="hexagonv66" "target-features"="+hvx-length64b,+hvxv66,+v66,-long-calls" }
attributes #1 = { "use-soft-float"="false" }
