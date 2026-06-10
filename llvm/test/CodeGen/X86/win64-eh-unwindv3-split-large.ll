; RUN: llc -mtriple=x86_64-unknown-windows-msvc \
; RUN:   -x86-wineh-unwindv3-epilog-distance-threshold=1 -o - %s | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-windows-msvc \
; RUN:   -x86-wineh-unwindv3-epilog-distance-threshold=1 -filetype=obj %s -o - \
; RUN:   | llvm-readobj --unwind - | FileCheck %s --check-prefix=OBJ

; Test V3 *size-based* sub-fragment splitting (the "Unwind v2 style" heuristic).
; With a very small distance threshold, every epilog is forced into its own
; chained sub-fragment, even though the function has far fewer than 8 epilogs.
; This exercises the path that keeps each tail-relative EpilogOffset within the
; signed 16-bit field for large functions.

declare i32 @c(i32) local_unnamed_addr

; CHECK-LABEL: three_epilogs:
; CHECK:         .seh_endprologue
; First epilog stays in the main fragment.
; CHECK:         .seh_startepilogue
; CHECK:         .seh_endepilogue
; A size-based split is inserted after each epilog.
; CHECK:         .seh_splitchained
; CHECK-NEXT:    .seh_endprologue
; CHECK:         .seh_startepilogue
; CHECK:         .seh_endepilogue
; CHECK:         .seh_splitchained
; CHECK-NEXT:    .seh_endprologue
; CHECK:         .seh_startepilogue
; CHECK:         .seh_endepilogue
; CHECK:         .seh_endproc

; Each fragment carries a single epilog with a small, in-range, tail-relative
; (negative) EpilogOffset, and every chained fragment is marked CHAININFO.
; OBJ:        Version: 3
; OBJ:        NumberOfEpilogs: 1
; OBJ:          EpilogOffset: -0x5
; OBJ:      RuntimeFunction {
; OBJ:        Version: 3
; OBJ:          ChainInfo (0x4)
; OBJ:        NumberOfEpilogs: 1
; OBJ:          EpilogOffset: -0x5

define dso_local i32 @three_epilogs(i32 %x) #0 {
entry:
  switch i32 %x, label %sw.default [
    i32 0, label %sw.0
    i32 1, label %sw.1
  ]
sw.0:
  %r0 = call i32 @c(i32 0)
  ret i32 %r0
sw.1:
  %r1 = call i32 @c(i32 1)
  ret i32 %r1
sw.default:
  %rd = call i32 @c(i32 7)
  ret i32 %rd
}

attributes #0 = { optnone noinline }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"winx64-eh-unwind", i32 3}
