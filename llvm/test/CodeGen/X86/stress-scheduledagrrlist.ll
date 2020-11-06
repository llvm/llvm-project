; RUN: llc -O0 -mtriple=x86_64-apple-macosx %s -o %t.s

; Stress test for the list scheduler. The store will be expanded to a very
; large number of stores during isel, stressing ScheduleDAGRRList. It should
; compiles in a reasonable amount of time. Run with -O0, to disable most other
; optimizations.

define void @test(i1000000* %ptr) {
entry:
  store i1000000 0, i1000000* %ptr, align 4
  ret void
}
