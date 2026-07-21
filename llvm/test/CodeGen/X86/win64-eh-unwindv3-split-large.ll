; RUN: llc -mtriple=x86_64-unknown-windows-msvc \
; RUN:   -x86-wineh-unwindv3-instr-avg-size=100000 -o - %s | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-windows-msvc \
; RUN:   -x86-wineh-unwindv3-instr-avg-size=100000 -filetype=obj %s -o - \
; RUN:   | llvm-readobj --unwind - | FileCheck %s --check-prefix=OBJ

; Test V3 *size-based* sub-fragment splitting (the "Unwind v2 style" heuristic).
; With a very small distance threshold, every epilog is forced into its own
; chained sub-fragment, even though the function has far fewer than 8 epilogs.
; This exercises the path that keeps each tail-relative EpilogOffset within the
; signed 16-bit field for large functions.

declare i32 @c(i32)

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

; Each epilog ends up in its own fragment with a small, in-range, tail-relative
; (negative) EpilogOffset. The main fragment holds the prolog; each subsequent
; fragment is an epilog-only chained fragment, and the trailing code after the
; last epilog becomes a final epilog-free chained fragment.
; OBJ:      UnwindInformation [
; Main fragment: holds the prolog and the first epilog.
; OBJ:        RuntimeFunction {
; OBJ:          UnwindInfo {
; OBJ-NEXT:       Version: 3
; OBJ-NEXT:       Flags [ (0x0)
; OBJ-NEXT:       ]
; OBJ:            NumberOfOps: 1
; OBJ-NEXT:       NumberOfEpilogs: 1
; OBJ-NEXT:       Prolog [1 ops]:
; OBJ-NEXT:         [0] IP +0x0000: ALLOC_SMALL Size=0x28
; OBJ-NEXT:       Epilog [0] {
; OBJ:              EpilogOffset: -0x5
; OBJ-NEXT:         NumberOfOps: 1
; OBJ-NEXT:         FirstOp: 0x0
; OBJ-NEXT:         IpOffsetOfLastInstruction: 0x4
; OBJ-NEXT:         [0] IP +0x0000: ALLOC_SMALL Size=0x28
; OBJ-NEXT:       }
; Second epilog: its own chained fragment (inherits the prolog from the parent).
; OBJ:        RuntimeFunction {
; OBJ:          UnwindInfo {
; OBJ-NEXT:       Version: 3
; OBJ-NEXT:       Flags [ (0x4)
; OBJ-NEXT:         ChainInfo (0x4)
; OBJ-NEXT:       ]
; OBJ:            NumberOfOps: 0
; OBJ-NEXT:       NumberOfEpilogs: 1
; OBJ-NEXT:       Prolog [0 ops]:
; OBJ-NEXT:       Epilog [0] {
; OBJ:              EpilogOffset: -0x5
; OBJ-NEXT:         NumberOfOps: 1
; OBJ-NEXT:         FirstOp: 0x0
; OBJ-NEXT:         IpOffsetOfLastInstruction: 0x4
; OBJ-NEXT:         [0] IP +0x0000: ALLOC_SMALL Size=0x28
; OBJ-NEXT:       }
; OBJ:            Chained {
; Third epilog: another chained fragment.
; OBJ:        RuntimeFunction {
; OBJ:          UnwindInfo {
; OBJ-NEXT:       Version: 3
; OBJ-NEXT:       Flags [ (0x4)
; OBJ-NEXT:         ChainInfo (0x4)
; OBJ-NEXT:       ]
; OBJ:            NumberOfEpilogs: 1
; OBJ:              EpilogOffset: -0x5
; OBJ:            Chained {
; Trailing code after the last epilog: an epilog-free chained fragment.
; OBJ:        RuntimeFunction {
; OBJ:          UnwindInfo {
; OBJ-NEXT:       Version: 3
; OBJ-NEXT:       Flags [ (0x4)
; OBJ-NEXT:         ChainInfo (0x4)
; OBJ-NEXT:       ]
; OBJ:            NumberOfOps: 0
; OBJ-NEXT:       NumberOfEpilogs: 0
; OBJ-NEXT:       Prolog [0 ops]:
; OBJ-NEXT:       Chained {

define i32 @three_epilogs(i32 %x) #0 {
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
