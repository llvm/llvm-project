; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: {{.balign 4|.p2align 2}}
; CHECK: {{.balign 4|.p2align 2}}

target triple = "hexagon"

; Function Attrs: nounwind optsize readnone
define i32 @f0() #0 section ".mysection.main" {
b0:
  ret i32 0
}

; Function Attrs: nounwind optsize readnone
define i32 @f1() #0 section ".mysection.anothermain" {
b0:
  ret i32 0
}

attributes #0 = { nounwind optsize readnone "target-cpu"="hexagonv55" }
