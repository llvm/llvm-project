; RUN: llc -O0 -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=MIR
; RUN: llc -O0 %s -o - | FileCheck %s --check-prefix=ASM

;; A loop header's PHI nodes are set up in its predecessor. When every incoming
;; value is a constant, that materialization was left without a source location,
;; so it silently inherited the line of whatever preceded it and the loop's own
;; line got its first is_stmt entry inside the header. The header is also the
;; back edge target, so a breakpoint on the loop's line was hit once per
;; iteration. This is the shape flang emits for a whole-array assignment, see
;; the source sketch next to the metadata below.
;;
;; Each statement here sits on a line of its own, so the checks pin down which
;; line is selected: the materialization has to take the location of the branch
;; entering the loop, and neither the preceding statement's nor the body's.

;; Bind the branch into each loop, and the statement ahead of it, to their
;; metadata nodes. The MIR checks then require the materializations to carry
;; exactly the branch's location rather than a line number that merely happens
;; to look plausible.
; MIR: store i32 7, ptr %arr, align 4, !dbg ![[LOCAL_BEFORE:[0-9]+]]
; MIR: br label %header, !dbg ![[LOCAL_INTO:[0-9]+]]
; MIR: store i32 7, ptr %array, align 4, !dbg ![[ARG_BEFORE:[0-9]+]]
; MIR: br label %loop, !dbg ![[ARG_INTO:[0-9]+]]
; MIR: call void @sink(ptr %array), !dbg ![[CALL_BEFORE:[0-9]+]]
; MIR: br label %loop, !dbg ![[CALL_INTO:[0-9]+]]

target triple = "x86_64-unknown-linux-gnu"

;; A loop the function falls into, with a constant for each of the two PHIs.
; MIR-LABEL: name: fill_local_
; MIR:        bb.0 (%ir-block.0):
; MIR:          MOV32mi %stack.0.arr, 1, $noreg, 0, $noreg, 7, debug-location ![[LOCAL_BEFORE]]
;; Only the first local value is given a location; the ones after it inherit it
;; through the line table.
; MIR-NEXT:     %[[LOCAL_I:[0-9]+]]:gr64 = MOV32ri64 1, debug-location ![[LOCAL_INTO]]
; MIR-NEXT:     %[[LOCAL_TRIP:[0-9]+]]:gr64 = MOV32ri64 4{{$}}
; MIR:        bb.1.header:
; MIR:          %{{[0-9]+}}:gr64_nosp = PHI %[[LOCAL_I]], %bb.0,
; MIR-NEXT:     %{{[0-9]+}}:gr64 = PHI %[[LOCAL_TRIP]], %bb.0,

; ASM-LABEL: fill_local_:
; ASM:         .loc 0 3 3 prologue_end
; ASM-NEXT:    movl $7,
; ASM:         .loc 0 4 3
; ASM-NEXT:    movl $1,
; ASM-NEXT:    movl $4,
;; The header only reaches its own line past the label, so a breakpoint on
;; line 4 binds in the predecessor and is reported once.
; ASM:       .LBB0_1:
; ASM:         .loc 0 5 5 is_stmt 1
; ASM-NEXT:    cmpq

define void @fill_local_() !dbg !5 {
  %arr = alloca [4 x i32], align 4
  store i32 7, ptr %arr, align 4, !dbg !10
  br label %header, !dbg !11

header:
  %i = phi i64 [ 1, %0 ], [ %i.next, %body ]
  %trip = phi i64 [ 4, %0 ], [ %trip.next, %body ]
  %more = icmp sgt i64 %trip, 0, !dbg !12
  br i1 %more, label %body, label %exit, !dbg !12

body:
  %offset = add nsw i64 %i, -1, !dbg !13
  %element = getelementptr i32, ptr %arr, i64 %offset, !dbg !13
  store i32 11, ptr %element, align 4, !dbg !13
  %i.next = add nsw i64 %i, 1, !dbg !13
  %trip.next = sub i64 %trip, 1, !dbg !13
  br label %header, !dbg !13

exit:
  call void @sink(ptr %arr), !dbg !14
  ret void, !dbg !15
}

;; The same through a dummy argument, which is the shape left once the loop has
;; been rotated and the induction variable starts from a negative constant.
; MIR-LABEL: name: fill_arg_
; MIR:        bb.0.entry:
; MIR:          MOV32mi %{{[0-9]+}}, 1, $noreg, 0, $noreg, 7, debug-location ![[ARG_BEFORE]]
; MIR-NEXT:     %[[ARG_I:[0-9]+]]:gr64 = MOV64ri32 -4, debug-location ![[ARG_INTO]]
; MIR:        bb.1.loop:
; MIR:          %{{[0-9]+}}:gr64_nosp = PHI %[[ARG_I]], %bb.0,

; ASM-LABEL: fill_arg_:
; ASM:         .loc 0 12 3 prologue_end
; ASM-NEXT:    movl $7,
; ASM:         .loc 0 13 3
; ASM-NEXT:    movq $-4,
; ASM:       .LBB1_1:
; ASM:         .loc 0 14 5 is_stmt 1
; ASM-NEXT:    movl $11,

define void @fill_arg_(ptr noalias writeonly captures(none) %array) !dbg !20 {
entry:
  store i32 7, ptr %array, align 4, !dbg !21
  br label %loop, !dbg !22

loop:
  %index = phi i64 [ -4, %entry ], [ %next, %loop ]
  %element = getelementptr i32, ptr %array, i64 %index, !dbg !23
  store i32 11, ptr %element, align 4, !dbg !23
  %next = add nuw nsw i64 %index, 1, !dbg !23
  %done = icmp eq i64 %next, 0, !dbg !23
  br i1 %done, label %exit, label %loop, !dbg !23

exit:
  ret void, !dbg !24
}

;; A loop further into the function, past a call. prologue_end says nothing
;; about this one, so the entry for the loop's line has to land ahead of it on
;; its own.
; MIR-LABEL: name: fill_after_call_
; MIR:        bb.0.entry:
; MIR:          CALL64pcrel32 target-flags(x86-plt) @sink, {{.*}} debug-location ![[CALL_BEFORE]]
; MIR:          %[[CALL_I:[0-9]+]]:gr64 = MOV64ri32 -4, debug-location ![[CALL_INTO]]
; MIR:        bb.1.loop:
; MIR:          %{{[0-9]+}}:gr64_nosp = PHI %[[CALL_I]], %bb.0,

; ASM-LABEL: fill_after_call_:
; ASM:         .loc 0 19 3 prologue_end
; ASM-NEXT:    callq sink
; ASM:         .loc 0 20 3
; ASM-NEXT:    movq $-4,
; ASM:       .LBB2_1:
; ASM:         .loc 0 21 5 is_stmt 1
; ASM-NEXT:    movl $11,

define void @fill_after_call_(ptr noalias writeonly captures(none) %array) !dbg !30 {
entry:
  call void @sink(ptr %array), !dbg !31
  br label %loop, !dbg !32

loop:
  %index = phi i64 [ -4, %entry ], [ %next, %loop ]
  %element = getelementptr i32, ptr %array, i64 %index, !dbg !33
  store i32 11, ptr %element, align 4, !dbg !33
  %next = add nuw nsw i64 %index, 1, !dbg !33
  %done = icmp eq i64 %next, 0, !dbg !33
  br i1 %done, label %exit, label %loop, !dbg !33

exit:
  ret void, !dbg !34
}

declare void @sink(ptr)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !1,
                             producer: "flang", isOptimized: false,
                             runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "assign.f90", directory: "/")
!2 = !DISubroutineType(types: !{})
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}

;; fill_local_ stands for
;;   1  subroutine fill_local()
;;   2    integer :: arr(4)
;;   3    arr(1) = 7
;;   4    arr = 11
;;   7    call sink(arr)
;;   8  end subroutine
;; except that the loop line 4 expands into is given lines 5 and 6 of its own,
;; so that a location taken from the loop is told apart from one taken from the
;; assignment. The other two subprograms are laid out the same way.
!5 = distinct !DISubprogram(name: "fill_local", linkageName: "fill_local_",
                            scope: !1, file: !1, line: 1, type: !2,
                            scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DILocation(line: 3, column: 3, scope: !5)
!11 = !DILocation(line: 4, column: 3, scope: !5)
!12 = !DILocation(line: 5, column: 5, scope: !5)
!13 = !DILocation(line: 6, column: 7, scope: !5)
!14 = !DILocation(line: 7, column: 3, scope: !5)
!15 = !DILocation(line: 8, column: 1, scope: !5)

!20 = distinct !DISubprogram(name: "fill_arg", linkageName: "fill_arg_",
                             scope: !1, file: !1, line: 10, type: !2,
                             scopeLine: 10, spFlags: DISPFlagDefinition, unit: !0)
!21 = !DILocation(line: 12, column: 3, scope: !20)
!22 = !DILocation(line: 13, column: 3, scope: !20)
!23 = !DILocation(line: 14, column: 5, scope: !20)
!24 = !DILocation(line: 15, column: 1, scope: !20)

!30 = distinct !DISubprogram(name: "fill_after_call", linkageName: "fill_after_call_",
                             scope: !1, file: !1, line: 17, type: !2,
                             scopeLine: 17, spFlags: DISPFlagDefinition, unit: !0)
!31 = !DILocation(line: 19, column: 3, scope: !30)
!32 = !DILocation(line: 20, column: 3, scope: !30)
!33 = !DILocation(line: 21, column: 5, scope: !30)
!34 = !DILocation(line: 22, column: 1, scope: !30)
