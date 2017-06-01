; RUN: opt -inline -S < %s | FileCheck %s
; RUN: opt -passes='cgscc(inline)' -S < %s | FileCheck %s

; Make sure the inliner doesn't crash when a metadata-bridged SSA operand is an
; undominated use.
;
; If we ever add a verifier check to prevent the scenario in this file, it's
; fine to delete this testcase.  However, we would need a bitcode upgrade since
; such historical IR exists in practice.

define i32 @foo(i32 %i) !dbg !4 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %add, i64 0, metadata !8, metadata !10), !dbg !11
  %add = add nsw i32 1, %i, !dbg !12
  ret i32 %add, !dbg !13
}

; CHECK-LABEL: define i32 @caller(
define i32 @caller(i32 %i) !dbg !3 {
; CHECK-NEXT: entry:
entry:
; Although the inliner shouldn't crash, it can't be expected to get the
; "correct" SSA value since its assumptions have been violated.
; CHECK-NEXT:   tail call void @llvm.dbg.value(metadata ![[EMPTY:[0-9]+]],
; CHECK-NEXT:   %{{.*}} = add nsw
  %call = tail call i32 @foo(i32 %i), !dbg !14
  ret i32 %call
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 (trunk 265634) (llvm/trunk 265637)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.c", directory: "/path/to/tests")
; CHECK: ![[EMPTY]] = !{}
!2 = !{}
!3 = distinct !DISubprogram(name: "caller", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0)
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !DILocalVariable(name: "add", arg: 1, scope: !4, file: !1, line: 2, type: !7)
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !DIExpression()
!11 = !DILocation(line: 2, column: 13, scope: !4)
!12 = !DILocation(line: 2, column: 27, scope: !4)
!13 = !DILocation(line: 2, column: 18, scope: !4)
!14 = !DILocation(line: 3, scope: !3)
