; RUN: llc -mtriple=hexagon -no-integrated-as < %s | FileCheck %s

; Check that constraints q and v are handled correctly.
; CHECK: q{{.}} = vgtw(v{{.}}.w,v{{.}}.w)
; CHECK: vand
; CHECK: vmem

target triple = "hexagon"

; Function Attrs: nounwind
define void @foo(<16 x i32> %v0, <16 x i32> %v1, ptr nocapture %p) #0 {
entry:
  %0 = tail call <64 x i1> asm "$0 = vgtw($1.w,$2.w)", "=q,v,v"(<16 x i32> %v0, <16 x i32> %v1) #1
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %0, i32 -1) #1
  store <16 x i32> %1, ptr %p, align 64
  ret void
}

declare <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1>, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
