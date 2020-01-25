; RUN: llc -mtriple=arm64-apple-ios -filetype=obj < %s -o %t.o
; RUN: llvm-dwarfdump %t.o | FileCheck %s -implicit-check-not=DW_TAG_subprogram

; The declaration subprogram for "function" is not in the CU's list of
; retained types. Test that a DWARF call site entry can still be constructed.

; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_name {{.*}}__hidden#3_
; CHECK:   DW_TAG_call_site
; CHECK:     DW_AT_call_origin (0x{{0+}}[[FUNCTION_DIE:.*]])

; CHECK: 0x{{0+}}[[FUNCTION_DIE]]: DW_TAG_subprogram
; CHECK:   DW_AT_name {{.*}}function

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios9.0.0"

define i32 @main() local_unnamed_addr !dbg !8 {
  %1 = tail call [2 x i64] @function([2 x i64] zeroinitializer), !dbg !11
  %2 = extractvalue [2 x i64] %1, 0, !dbg !11
  %3 = trunc i64 %2 to i32, !dbg !11
  ret i32 %3, !dbg !12
}

declare !dbg !13 [2 x i64] @function([2 x i64]) local_unnamed_addr

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.dbg.cu = !{!5}
!llvm.ident = !{!7}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 13, i32 4]}
!1 = !{i32 7, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"PIC Level", i32 2}
!5 = distinct !DICompileUnit(language: DW_LANG_C99, file: !6, producer: "__hidden#0_", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, nameTableKind: None)
!6 = !DIFile(filename: "__hidden#1_", directory: "__hidden#2_")
!7 = !{!"Apple clang version 11.0.0 (llvm-project fa407d93fd5e618d76378c1ce4e4f517e0563278) (+internal-os)"}
!8 = distinct !DISubprogram(name: "__hidden#3_", scope: !6, file: !6, line: 9, type: !9, scopeLine: 9, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !5)
!9 = !DISubroutineType(types: !10)
!10 = !{}
!11 = !DILocation(line: 12, column: 10, scope: !8)
!12 = !DILocation(line: 13, column: 3, scope: !8)
!13 = !DISubprogram(name: "function", scope: !6, file: !6, line: 7, type: !9, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
