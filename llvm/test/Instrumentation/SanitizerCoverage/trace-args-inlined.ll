; An inlined callee's parameters are numbered just like the caller's, and the
; records describing them may name the caller's own arguments. Only records
; belonging to the function being instrumented may drive trace-args, otherwise
; an inlined callee's first parameter would be reported as the caller's.
;
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-trace-args -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; void caller(long size) calling an inlined callee(long n): the inlined
; parameter `n` is also arg 1 and is described by a record naming %size, but it
; belongs to @callee, so only @caller's own record for `size` is used.
define void @caller(i64 %size, ptr %p) !dbg !6 {
entry:
    #dbg_value(i64 %size, !14, !DIExpression(), !16)
    #dbg_value(ptr %p, !15, !DIExpression(), !16)
    #dbg_value(i64 %size, !12, !DIExpression(), !17)
  ret void
}
; Both of @caller's parameters are reported from their own records. Nothing is
; reported for the inlined `n`.
; CHECK-LABEL: define void @caller(
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @caller to i64), i32 0, i32 8, i64 %size, ptr null, i32 0)
; CHECK: %[[P:[0-9]+]] = ptrtoint ptr %p to i64
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @caller to i64), i32 1, i32 8, i64 %[[P]], ptr null, i32 0)
; CHECK-NOT: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @caller to i64), i32 2

declare void @callee(i64)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, isOptimized: true, emissionKind: FullDebug)
!1 = !{i32 2, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "inlined.c", directory: "/")
!4 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64)

!6 = distinct !DISubprogram(name: "caller", scope: !3, file: !3, line: 10, type: !7, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !4, !5}
!9 = distinct !DISubprogram(name: "callee", scope: !3, file: !3, line: 1, type: !10, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!10 = !DISubroutineType(types: !{null, !4})
!11 = !{!12}
!12 = !DILocalVariable(name: "n", arg: 1, scope: !9, file: !3, line: 1, type: !4)
!13 = !{!14, !15}
!14 = !DILocalVariable(name: "size", arg: 1, scope: !6, file: !3, line: 10, type: !4)
!15 = !DILocalVariable(name: "p", arg: 2, scope: !6, file: !3, line: 10, type: !5)
!16 = !DILocation(line: 10, column: 1, scope: !6)
!17 = !DILocation(line: 1, column: 1, scope: !9, inlinedAt: !16)
