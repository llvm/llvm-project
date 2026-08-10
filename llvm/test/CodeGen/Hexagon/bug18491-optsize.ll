; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK: {{.balign 4|.p2align 2}}
; CHECK: {{.balign 4|.p2align 2}}
; CHECK: {{.balign 4|.p2align 2}}

target triple = "hexagon"

@g0 = global i32 4, align 4
@g1 = global i32 4, align 4
@g2 = global i32 4, align 4
@g3 = global i32 4, align 4

; Function Attrs: nounwind optsize
define void @f0(i32 %a0) #0 {
b0:
  store i32 1, ptr @g0, align 4
  ret void
}

; Function Attrs: nounwind optsize
define void @f1(i32 %a0) #0 {
b0:
  store i32 1, ptr @g0, align 4
  store i32 2, ptr @g1, align 4
  store i32 3, ptr @g2, align 4
  store i32 4, ptr @g3, align 4
  ret void
}

; Function Attrs: nounwind optsize readnone
define i32 @f2(i32 %a0, ptr nocapture readnone %a1) #1 {
b0:
  ret i32 %a0
}

attributes #0 = { nounwind optsize "target-cpu"="hexagonv60" }
attributes #1 = { nounwind optsize readnone "target-cpu"="hexagonv60" }
