; RUN: llc -O0 %s -o - | FileCheck %s

;; A loop header's PHI nodes are set up in the predecessor. When every incoming
;; value is a constant, that code used to be left without a source location, so
;; the loop's line got its first is_stmt entry inside the header. The header is
;; also the target of the back edge, so a breakpoint on that line was hit once
;; per iteration. This is the shape flang emits for a whole-array assignment,
;;   integer :: arr(4)
;;   arr = 11

;; A loop the function falls into: everything from the first block label onwards
;; belongs to it, so prologue_end has to be reached before one is seen.
; CHECK-LABEL: fill_local_:
; CHECK-NOT:   {{^\.LBB}}
; CHECK:       .loc {{[0-9]+}} 3 3 prologue_end

;; The same through a dummy argument, which reproduces once the loop has been
;; rotated and the induction variable starts from a negative constant.
; CHECK-LABEL: fill_arg_:
; CHECK-NOT:   {{^\.LBB}}
; CHECK:       .loc {{[0-9]+}} 8 3 prologue_end

;; A loop further into the function, past a call. prologue_end says nothing
;; about this one, so the line-table entry itself has to land ahead of the loop.
; CHECK-LABEL: fill_after_call_:
; CHECK-NOT:   {{^\.LBB}}
; CHECK:       .loc {{[0-9]+}} 13 3

target triple = "x86_64-unknown-linux-gnu"

define void @fill_local_() !dbg !5 {
  %arr = alloca [4 x i32], align 4
  br label %header, !dbg !10

header:
  %i = phi i64 [ 1, %0 ], [ %i.next, %body ]
  %trip = phi i64 [ 4, %0 ], [ %trip.next, %body ]
  %more = icmp sgt i64 %trip, 0, !dbg !10
  br i1 %more, label %body, label %exit, !dbg !10

body:
  %offset = add nsw i64 %i, -1, !dbg !10
  %element = getelementptr i32, ptr %arr, i64 %offset, !dbg !10
  store i32 11, ptr %element, align 4, !dbg !10
  %i.next = add nsw i64 %i, 1, !dbg !10
  %trip.next = sub i64 %trip, 1, !dbg !10
  br label %header, !dbg !10

exit:
  call void @sink(ptr %arr), !dbg !11
  ret void, !dbg !11
}

define void @fill_arg_(ptr noalias writeonly captures(none) %array) !dbg !12 {
entry:
  br label %loop, !dbg !13

loop:
  %index = phi i64 [ -4, %entry ], [ %next, %loop ]
  %element = getelementptr i32, ptr %array, i64 %index, !dbg !13
  store i32 11, ptr %element, align 4, !dbg !13
  %next = add nuw nsw i64 %index, 1, !dbg !13
  %done = icmp eq i64 %next, 0, !dbg !13
  br i1 %done, label %exit, label %loop, !dbg !13

exit:
  ret void, !dbg !14
}

define void @fill_after_call_(ptr noalias writeonly captures(none) %array) !dbg !15 {
entry:
  call void @sink(ptr %array), !dbg !16
  br label %loop, !dbg !17

loop:
  %index = phi i64 [ -4, %entry ], [ %next, %loop ]
  %element = getelementptr i32, ptr %array, i64 %index, !dbg !17
  store i32 11, ptr %element, align 4, !dbg !17
  %next = add nuw nsw i64 %index, 1, !dbg !17
  %done = icmp eq i64 %next, 0, !dbg !17
  br i1 %done, label %exit, label %loop, !dbg !17

exit:
  ret void, !dbg !18
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
!5 = distinct !DISubprogram(name: "fill_local", linkageName: "fill_local_",
                            scope: !1, file: !1, line: 1, type: !2,
                            scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DILocation(line: 3, column: 3, scope: !5)
!11 = !DILocation(line: 4, column: 1, scope: !5)
!12 = distinct !DISubprogram(name: "fill_arg", linkageName: "fill_arg_",
                             scope: !1, file: !1, line: 6, type: !2,
                             scopeLine: 6, spFlags: DISPFlagDefinition, unit: !0)
!13 = !DILocation(line: 8, column: 3, scope: !12)
!14 = !DILocation(line: 9, column: 1, scope: !12)
!15 = distinct !DISubprogram(name: "fill_after_call", linkageName: "fill_after_call_",
                             scope: !1, file: !1, line: 11, type: !2,
                             scopeLine: 11, spFlags: DISPFlagDefinition, unit: !0)
!16 = !DILocation(line: 12, column: 3, scope: !15)
!17 = !DILocation(line: 13, column: 3, scope: !15)
!18 = !DILocation(line: 14, column: 1, scope: !15)
