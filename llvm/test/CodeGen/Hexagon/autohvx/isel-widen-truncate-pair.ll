; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; This has a v32i8 = truncate v16i32 (64b mode), which was legalized to
; 64i8 = vpackl v32i32, for which there were no selection patterns provided.
; Check that we generate vpackeh->vpackeb for this.

; CHECK-LABEL: fred:
; CHECK: v[[V0:[0-9]+]].h = vpacke(v1.w,v0.w)
; CHECK:                  = vpacke({{.*}},v[[V0]].h)
define void @fred(ptr %a0, <32 x i32> %a1) #0 {
  %v0 = trunc <32 x i32> %a1 to <32 x i8>
  store <32 x i8> %v0, ptr %a0, align 32
  ret void
}

attributes #0 = { "target-cpu"="hexagonv65" "target-features"="+hvx,+hvx-length64b" }

