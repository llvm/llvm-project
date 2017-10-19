; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that constraints q and v are handled correctly.
; CHECK: q{{.}} = vgtw(v{{.}}.w,v{{.}}.w)
; CHECK: vand
; CHECK: vmem

target triple = "hexagon"

; Function Attrs: nounwind
define void @foo(<16 x i32> %v0, <16 x i32> %v1, <16 x i32>* nocapture %p) #0 {
entry:
  %0 = tail call <16 x i32> asm "$0 = vgtw($1.w,$2.w)", "=q,v,v"(<16 x i32> %v0, <16 x i32> %v1) #1
  store <16 x i32> %0, <16 x i32>* %p, align 64
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" } 
attributes #1 = { nounwind readnone }
