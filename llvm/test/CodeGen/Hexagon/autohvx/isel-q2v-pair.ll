; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Make sure that this doesn't crash.
; CHECK: vadd

define void @foo(ptr %a0, ptr %a1) #0 {
  %v0 = load <64 x i32>, ptr %a0, align 128
  %v1 = load <64 x i32>, ptr %a1, align 128
  %v2 = icmp sgt <64 x i32> %v0, zeroinitializer
  %v3 = sext <64 x i1> %v2 to <64 x i32>
  %v4 = add nsw <64 x i32> %v1, %v3
  store <64 x i32> %v4, ptr %a1, align 128
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv65" "target-features"="+hvx,+hvx-length128b" }
