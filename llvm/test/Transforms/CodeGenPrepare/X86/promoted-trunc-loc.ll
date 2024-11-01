; RUN: opt < %s -codegenprepare -S -mtriple=x86_64-unknown-unknown | FileCheck %s --match-full-lines

; Make sure the promoted trunc doesn't get a debug location associated.
; CHECK: %promoted = trunc i32 %or to i16

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

@b = global i16 0, align 2

declare void @bar(i64)

define i32 @foo(i16 %kkk) !dbg !6 {
entry:
  %t4 = load i16, ptr @b, align 2, !dbg !8
  %conv4 = zext i16 %t4 to i32, !dbg !9
  %or = or i16 %kkk, %t4, !dbg !10
  %c = sext i16 %or to i64, !dbg !11
  call void @bar(i64 %c), !dbg !12
  %t5 = and i16 %or, 5, !dbg !13
  %z = zext i16 %t5 to i32, !dbg !14
  ret i32 %z, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "/Users/davide/work/llvm-project/patatino.ll", directory: "/")
!2 = !{}
!3 = !{i32 8}
!4 = !{i32 0}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 1, column: 1, scope: !6)
!9 = !DILocation(line: 2, column: 1, scope: !6)
!10 = !DILocation(line: 3, column: 1, scope: !6)
!11 = !DILocation(line: 4, column: 1, scope: !6)
!12 = !DILocation(line: 5, column: 1, scope: !6)
!13 = !DILocation(line: 6, column: 1, scope: !6)
!14 = !DILocation(line: 7, column: 1, scope: !6)
!15 = !DILocation(line: 8, column: 1, scope: !6)
