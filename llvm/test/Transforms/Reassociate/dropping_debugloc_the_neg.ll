; RUN: opt < %s -passes=reassociate -S | FileCheck %s

; Check that Reassociate's NegateValue() drops the debug location of `%sub2` 
; when moving it out of the loop's body (`%for.body`) to `%for.cond`.

define void @fn1(i32 %a, i1 %c, ptr %ptr) !dbg !5 {
; CHECK-LABEL: define void @fn1(
; CHECK:       for.cond:
; CHECK:        [[SUB2:%.*]] = sub i32 0, %d.0{{$}}
; CHECK:       for.body:
;
entry:
  br label %for.cond, !dbg !8

for.cond:                                         ; preds = %for.body, %entry
  %d.0 = phi i32 [ 1, %entry ], [ 2, %for.body ], !dbg !9
  br i1 %c, label %for.end, label %for.body, !dbg !10

for.body:                                         ; preds = %for.cond
  %sub1 = sub i32 %a, %d.0, !dbg !11
  %dead1 = add i32 %sub1, 1, !dbg !12
  %dead2 = mul i32 %dead1, 3, !dbg !13
  %dead3 = mul i32 %dead2, %sub1, !dbg !14
  %sub2 = sub nsw i32 0, %d.0, !dbg !15
  store i32 %sub2, ptr %ptr, align 4, !dbg !16
  br label %for.cond, !dbg !17

for.end:                                          ; preds = %for.cond
  ret void, !dbg !18
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "debugloc.ll", directory: "/")
!2 = !{i32 11}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "fn1", linkageName: "fn1", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 1, column: 1, scope: !5)
!9 = !DILocation(line: 2, column: 1, scope: !5)
!10 = !DILocation(line: 3, column: 1, scope: !5)
!11 = !DILocation(line: 4, column: 1, scope: !5)
!12 = !DILocation(line: 5, column: 1, scope: !5)
!13 = !DILocation(line: 6, column: 1, scope: !5)
!14 = !DILocation(line: 7, column: 1, scope: !5)
!15 = !DILocation(line: 8, column: 1, scope: !5)
!16 = !DILocation(line: 9, column: 1, scope: !5)
!17 = !DILocation(line: 10, column: 1, scope: !5)
!18 = !DILocation(line: 11, column: 1, scope: !5)
