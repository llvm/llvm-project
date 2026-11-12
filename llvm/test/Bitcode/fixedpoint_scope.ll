;; This test checks generation of DIFixedPointType with scopes.

; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

;; Test whether DIFixedPointType is generated.
; CHECK: !DIFixedPointType(name: "fp__decimal", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 2, size: 32, align: 32, encoding: DW_ATE_signed_fixed, kind: Decimal, factor: -4)
; CHECK: !DIFixedPointType(name: "fp__rational", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 2, size: 32, align: 32, encoding: DW_ATE_unsigned_fixed, kind: Rational, numerator: 1234, denominator: 5678)
; CHECK: !DIFixedPointType(name: "fp__binary", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 2, size: 64, encoding: DW_ATE_unsigned_fixed, kind: Binary, factor: -16)

; ModuleID = 'fixedpoint_type.ll'
source_filename = "/dir/fixedpoint_type.adb"

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @foo()

define void @bar(ptr %this) !dbg !6 {
  call void @foo(), !dbg !14
  ret void, !dbg !14
}

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 2, !"Dwarf Version", i32 4}
!2 = distinct !DICompileUnit(language: DW_LANG_Ada95, file: !3, producer: "GNAT/LLVM", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!3 = !DIFile(filename: "fixedpoint_type.adb", directory: "/dir")

!6 = distinct !DISubprogram(name: "fp", scope: !3, file: !3, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !9)
!7 = !DISubroutineType(types: !8)
!8 = !{}
!9 = !{!10, !11, !12, !13}
!10 = !DILocalVariable(name: "x", scope: !6, file: !3, line: 3, type: !11, align: 32)
!11 = !DIFixedPointType(name: "fp__decimal", size: 32, align: 32, encoding: DW_ATE_signed_fixed, kind: Decimal, factor: -4, scope: !6, file: !3, line: 2)
!12 = !DIFixedPointType(name: "fp__rational", size: 32, align: 32, encoding: DW_ATE_unsigned_fixed, kind: Rational, numerator: 1234, denominator: 5678, scope: !6, file: !3, line: 2)
!13 = !DIFixedPointType(name: "fp__binary", size: 64, align: 0, encoding: DW_ATE_unsigned_fixed, kind: Binary, factor: -16, scope: !6, file: !3, line: 2)
!14 = !DILocation(line: 69, column: 18, scope: !6)
