; RUN: llc -mtriple=x86_64-unknown-windows-msvc -o - %s | FileCheck %s

; Test that the V3 pass splits a function with >7 epilogs into chained
; sub-fragments. The switch produces 8 cases, each with its own epilog.
; The pass should insert a .seh_splitchained before the 8th epilog.

declare i32 @c(i32) local_unnamed_addr

; CHECK-LABEL: eight_epilogs:
; CHECK:       .seh_proc eight_epilogs
; CHECK:       .seh_stackalloc 40
; CHECK:       .seh_endprologue

; Epilogs 1-7 in the main fragment.
; CHECK-COUNT-7: .seh_startepilogue

; Split before the 8th epilog.
; CHECK:       .seh_splitchained
; CHECK-NEXT:  .seh_endprologue

; The 8th epilog is in the chained fragment.
; CHECK:       .seh_startepilogue
; CHECK:       .seh_endepilogue
; CHECK:       .seh_endproc

define dso_local i32 @eight_epilogs(i32 %x) #0 {
entry:
  switch i32 %x, label %sw.default [
    i32 0, label %sw.0
    i32 1, label %sw.1
    i32 2, label %sw.2
    i32 3, label %sw.3
    i32 4, label %sw.4
    i32 5, label %sw.5
    i32 6, label %sw.6
  ]

sw.0:
  %r0 = call i32 @c(i32 0)
  ret i32 %r0
sw.1:
  %r1 = call i32 @c(i32 1)
  ret i32 %r1
sw.2:
  %r2 = call i32 @c(i32 2)
  ret i32 %r2
sw.3:
  %r3 = call i32 @c(i32 3)
  ret i32 %r3
sw.4:
  %r4 = call i32 @c(i32 4)
  ret i32 %r4
sw.5:
  %r5 = call i32 @c(i32 5)
  ret i32 %r5
sw.6:
  %r6 = call i32 @c(i32 6)
  ret i32 %r6
sw.default:
  %rd = call i32 @c(i32 7)
  ret i32 %rd
}

attributes #0 = { optnone noinline }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"winx64-eh-unwind", i32 3}
