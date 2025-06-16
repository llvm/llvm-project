; RUN: opt < %s -passes=always-inline -S | FileCheck %s
; Test that inlined allocas do not retain debug locations from the callee
; to prevent incorrect source attribution in generated code.

; ModuleID = 'test-inline-debug-alloca.c'
source_filename = "test-inline-debug-alloca.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: alwaysinline nounwind uwtable
define dso_local i32 @callee() #0 !dbg !7 {
entry:
  %large_array = alloca [1000 x i8], align 1, !dbg !12
  %x = alloca i32, align 4, !dbg !13
  %arrayidx = getelementptr inbounds [1000 x i8], ptr %large_array, i64 0, i64 0, !dbg !14
  store i8 42, ptr %arrayidx, align 1, !dbg !15
  store i32 123, ptr %x, align 4, !dbg !16
  %0 = load i32, ptr %x, align 4, !dbg !17
  ret i32 %0, !dbg !18
}

; Function Attrs: nounwind uwtable
define dso_local i32 @caller() #1 !dbg !19 {
entry:
  %call = call i32 @callee(), !dbg !20
  ret i32 %call, !dbg !21
}

; CHECK-LABEL: define dso_local i32 @caller()
; CHECK: entry:
; CHECK: %large_array.i = alloca [1000 x i8], align 1{{$}}
; CHECK-NOT: %large_array.i = alloca [1000 x i8], align 1, !dbg
; CHECK: %x.i = alloca i32, align 4{{$}}
; CHECK-NOT: %x.i = alloca i32, align 4, !dbg
; CHECK: ret i32

attributes #0 = { alwaysinline nounwind uwtable }
attributes #1 = { nounwind uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false)
!1 = !DIFile(filename: "test-inline-debug-alloca.c", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{!"clang"}
!7 = distinct !DISubprogram(name: "callee", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{}
!12 = !DILocation(line: 2, column: 9, scope: !7)
!13 = !DILocation(line: 3, column: 9, scope: !7)
!14 = !DILocation(line: 4, column: 3, scope: !7)
!15 = !DILocation(line: 4, column: 16, scope: !7)
!16 = !DILocation(line: 5, column: 5, scope: !7)
!17 = !DILocation(line: 6, column: 10, scope: !7)
!18 = !DILocation(line: 6, column: 3, scope: !7)
!19 = distinct !DISubprogram(name: "caller", scope: !1, file: !1, line: 9, type: !8, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !11)
!20 = !DILocation(line: 10, column: 10, scope: !19)
!21 = !DILocation(line: 10, column: 3, scope: !19)
