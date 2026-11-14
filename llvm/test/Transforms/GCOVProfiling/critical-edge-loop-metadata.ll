; NOTE: Do not autogenerate. Check loop metadata ownership across a batch of splits.
; RUN: mkdir -p %t && cd %t
; RUN: opt -S -verify-each -passes=insert-gcov-profiling %s | FileCheck %s

; Exercise splits before the first annotated loop and between annotated loops.
; The temporary dominator tree must include earlier splits and track later ones.
; CHECK-LABEL: define void @loops(
; CHECK: first:
; CHECK: br i1 %again, label %[[FIRST:[^, ]+]], label %second, !dbg ![[LOC:[0-9]+]]{{$}}
; CHECK: [[FIRST]]:
; CHECK: br label %first, !dbg ![[LOC]]{{$}}
; CHECK: second:
; CHECK: br i1 %again, label %[[SECOND:[^, ]+]], label %third, !dbg ![[LOC]]{{$}}
; CHECK: [[SECOND]]:
; CHECK: br label %second, !dbg ![[LOC]], !llvm.loop ![[LOOP:[0-9]+]]
; CHECK: third:
; CHECK: br i1 %again, label %[[THIRD:[^, ]+]], label %fourth, !dbg ![[LOC]]{{$}}
; CHECK: [[THIRD]]:
; CHECK: br label %third, !dbg ![[LOC]]{{$}}
; CHECK: fourth:
; CHECK: br i1 %again, label %[[FOURTH:[^, ]+]], label %exit, !dbg ![[LOC]]{{$}}
; CHECK: [[FOURTH]]:
; CHECK: br label %fourth, !dbg ![[LOC]], !llvm.loop ![[LOOP]]

define void @loops(i1 %again) !dbg !5 {
entry:
  br label %first, !dbg !8
first:
  br i1 %again, label %first, label %second, !dbg !8
second:
  br i1 %again, label %second, label %third, !dbg !8, !llvm.loop !9
third:
  br i1 %again, label %third, label %fourth, !dbg !8
fourth:
  br i1 %again, label %fourth, label %exit, !dbg !8, !llvm.loop !9
exit:
  ret void, !dbg !8
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "loops.c", directory: "")
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "loops", scope: !1, file: !1, line: 1, type: !6, spFlags: DISPFlagDefinition, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !11}
!8 = !DILocation(line: 2, column: 1, scope: !5)
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.unroll.disable"}
!11 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
