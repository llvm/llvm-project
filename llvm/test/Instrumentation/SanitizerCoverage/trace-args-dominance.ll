; A debug record may name a value that the trace call cannot use: debug records
; are exempt from SSA dominance, so the value may be defined after the call, and
; a parameter that interprocedural optimization proved dead is described as
; poison. Neither can be reported, and using the first would produce IR that
; fails the verifier with "Instruction does not dominate all uses". Such
; parameters are reported with size 0.
;
; opt runs the verifier, so a passing run also proves the emitted IR is
; well-formed.
;
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-trace-args -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; The only location of `a` is %later, defined in a block that the entry block
; branches to. `b` is a plain pointer argument and is reported normally.
define void @location_does_not_dominate(ptr %a, ptr %b) !dbg !6 {
entry:
    #dbg_value(ptr %later, !10, !DIExpression(), !12)
    #dbg_value(ptr %b, !11, !DIExpression(), !12)
  br label %bb, !dbg !12
bb:
  %later = getelementptr i8, ptr %a, i64 128, !dbg !12
  ret void, !dbg !12
}
; CHECK-LABEL: define void @location_does_not_dominate(
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @location_does_not_dominate to i64), i32 0, i32 0, i64 0, ptr null, i32 0)
; CHECK: %[[B:[0-9]+]] = ptrtoint ptr %b to i64
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @location_does_not_dominate to i64), i32 1, i32 8, i64 %[[B]], ptr null, i32 0)

; `a` is dead and described as poison; `b` is live.
define void @poison_location(ptr %b) !dbg !13 {
entry:
    #dbg_value(ptr poison, !15, !DIExpression(), !17)
    #dbg_value(ptr %b, !16, !DIExpression(), !17)
  ret void
}
; CHECK-LABEL: define void @poison_location(
; CHECK-NOT: ptrtoint ptr poison
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @poison_location to i64), i32 0, i32 0, i64 0, ptr null, i32 0)
; CHECK: %[[PB:[0-9]+]] = ptrtoint ptr %b to i64
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @poison_location to i64), i32 1, i32 8, i64 %[[PB]], ptr null, i32 0)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, isOptimized: true, emissionKind: FullDebug)
!1 = !{i32 2, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "dominance.c", directory: "/")
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64)

!6 = distinct !DISubprogram(name: "location_does_not_dominate", scope: !3, file: !3, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !9)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !5, !5}
!9 = !{!10, !11}
!10 = !DILocalVariable(name: "a", arg: 1, scope: !6, file: !3, line: 1, type: !5)
!11 = !DILocalVariable(name: "b", arg: 2, scope: !6, file: !3, line: 1, type: !5)
!12 = !DILocation(line: 1, column: 1, scope: !6)

!13 = distinct !DISubprogram(name: "poison_location", scope: !3, file: !3, line: 6, type: !7, scopeLine: 6, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!14 = !{!15, !16}
!15 = !DILocalVariable(name: "a", arg: 1, scope: !13, file: !3, line: 6, type: !5)
!16 = !DILocalVariable(name: "b", arg: 2, scope: !13, file: !3, line: 6, type: !5)
!17 = !DILocation(line: 6, column: 1, scope: !13)
