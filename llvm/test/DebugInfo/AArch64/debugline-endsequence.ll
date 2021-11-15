; RUN: llc %s -filetype=obj -o - | llvm-dwarfdump --debug-line - | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx12.0.0"

; Check if the end_sequences are emitted for each debug range.

; CU1 Line table
; CHECK: 0x0000000000000004 [[T:.*]] end_sequence
; CHECK: 0x0000000000000010 [[T:.*]] end_sequence
;
; CU2 Line table
; CHECK: 0x0000000000000008 [[T:.*]] end_sequence

; CU1 (0x0 ~ 0x4)
define void @f1() !dbg !15 {
  ret void, !dbg !18
}

; CU2 (0x4 ~ 0x8)
define void @f2() !dbg !21 {
  ret void, !dbg !22
}

; CU2 (nodebug) - (0x8 ~ 0xc)
define void @f3() {
  ret void
}

; CU1 (0xc ~ 0x10)
define void @f4() !dbg !19 {
  ret void, !dbg !20
}

!llvm.dbg.cu = !{!0, !3}
!llvm.ident = !{!5, !5}
!llvm.module.flags = !{!6, !7, !8, !9, !10, !11, !12, !13, !14}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "LLVM", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None, sysroot: "/")
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{}
!3 = distinct !DICompileUnit(language: DW_LANG_C99, file: !4, producer: "LLVM", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None, sysroot: "/")
!4 = !DIFile(filename: "<stdin>", directory: "/")
!5 = !{!"Apple clang version 13.0.0 (clang-1300.0.29.3)"}
!6 = !{i32 2, !"SDK Version", [2 x i32] [i32 11, i32 3]}
!7 = !{i32 7, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 1, !"branch-target-enforcement", i32 0}
!11 = !{i32 1, !"sign-return-address", i32 0}
!12 = !{i32 1, !"sign-return-address-all", i32 0}
!13 = !{i32 1, !"sign-return-address-with-bkey", i32 0}
!14 = !{i32 7, !"PIC Level", i32 2}
!15 = distinct !DISubprogram(name: "f1", scope: !1, file: !1, line: 1, type: !16, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!16 = !DISubroutineType(types: !17)
!17 = !{null}
!18 = !DILocation(line: 2, column: 1, scope: !15)
!19 = distinct !DISubprogram(name: "f4", scope: !1, file: !1, line: 4, type: !16, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!20 = !DILocation(line: 5, column: 1, scope: !19)
!21 = distinct !DISubprogram(name: "f2", scope: !4, file: !4, line: 1, type: !16, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !3, retainedNodes: !2)
!22 = !DILocation(line: 2, column: 1, scope: !21)
