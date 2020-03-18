; RUN: llc -mtriple aarch64-linux-gnu -emit-call-site-info -debug-entry-values -filetype=obj -o - %s | llvm-dwarfdump - | FileCheck %s

; Based on the following C reproducer:
;
; extern void callee(float);
;
; void foo(float param) {
;   callee(param);
; }

; Verify that a call site value using DW_OP_GNU_entry_value(DW_OP_regx B0) is
; emitted for the float parameter. Previously the entry value's multi-byte
; DW_OP_regx expression would be truncated.

; CHECK: DW_TAG_GNU_call_site_parameter
; CHECK-NEXT: DW_AT_location (DW_OP_regx B0)
; CHECK-NEXT: DW_AT_GNU_call_site_value (DW_OP_GNU_entry_value(DW_OP_regx B0)

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; Function Attrs: nounwind
define dso_local void @foo(float %param) local_unnamed_addr !dbg !12 {
entry:
  tail call void @callee(float %param), !dbg !13
  ret void, !dbg !14
}

declare !dbg !4 dso_local void @callee(float) local_unnamed_addr

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "float.c", directory: "/")
!2 = !{}
!3 = !{!4}
!4 = !DISubprogram(name: "callee", scope: !1, file: !1, line: 1, type: !5, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7}
!7 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!8 = !{i32 7, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{!"clang version 11.0.0 "}
!12 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !5, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!13 = !DILocation(line: 4, scope: !12)
!14 = !DILocation(line: 5, scope: !12)
