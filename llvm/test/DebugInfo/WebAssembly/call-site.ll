; RUN: llc -filetype=obj %s -o - | llvm-dwarfdump - | FileCheck %s

;; This checks if the call site information is correctly written in debug info.
;; This is a regression test for the bug that DwarfDebug unconditionally assumed
;; the callee operand was getOperand(0), which was not true for WebAssembly.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK: 0x00000026:   DW_TAG_subprogram
; CHECK:                 DW_AT_name  ("call_direct")
; CHECK: 0x0000003d:     DW_TAG_GNU_call_site
; CHECK-NEXT:              DW_AT_abstract_origin (0x00000047 "foo")

define i32 @call_direct() !dbg !6 {
entry:
  %0 = call i32 @foo(), !dbg !8
  ret i32 %0, !dbg !9
}

;; WebAssembly does not currently support DW_TAG_GNU_call_site for stackified
;; registers. This just checks if the test runs without crashing.
define i32 @call_indirect(ptr %callee) !dbg !11 {
  %1 = call i32 %callee(i32 3, i32 5), !dbg !12
  ret i32 %1, !dbg !13
}

declare !dbg !10 i32 @foo()
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/home/llvm-project")
!2 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = distinct !DISubprogram(name: "call_direct", scope: !1, file: !1, line: 3, type: !7, scopeLine: 3, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!7 = !DISubroutineType(types: !{null})
!8 = !DILocation(line: 4, column: 11, scope: !6)
!9 = !DILocation(line: 7, column: 1, scope: !6)
!10 = !DISubprogram(name: "foo", scope: !1, file: !1, line: 30, type: !7, scopeLine: 3, spFlags: DISPFlagOptimized)
!11 = distinct !DISubprogram(name: "call_indirect", scope: !1, file: !1, line: 3, type: !7, scopeLine: 3, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!12 = !DILocation(line: 40, column: 11, scope: !11)
!13 = !DILocation(line: 70, column: 1, scope: !11)
