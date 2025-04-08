; RUN: opt < %s -passes=inline -S | FileCheck %s

; This tests that functions with the attribute `no-inline-line-tables` have the
; correct debug information when they are inlined.

; ModuleID = 't.c'
source_filename = "t.c"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-msvc"

; Function Attrs: alwaysinline nounwind
define dso_local i32 @f(i32 %x) #0 !dbg !7 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %x.addr, metadata !12, metadata !DIExpression()), !dbg !13
  %0 = load i32, ptr %x.addr, align 4, !dbg !14
  ret i32 %0, !dbg !14
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: alwaysinline nounwind
define i32 @g(i32 %x) #0 !dbg !15 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %x.addr, metadata !16, metadata !DIExpression()), !dbg !17
  br label %L, !dbg !17

L:                                                ; preds = %entry
  call void @llvm.dbg.label(metadata !18), !dbg !19
  store i32 42, ptr %x.addr, align 4, !dbg !20
  %0 = load i32, ptr %x.addr, align 4, !dbg !21
  ret i32 %0, !dbg !21
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.label(metadata) #1

; Check that debug info for inlined code uses the call location and that debug
; intrinsics are removed.
; Function Attrs: noinline nounwind optnone
define i32 @main() #2 !dbg !22 {
entry:
; CHECK-LABEL: @main()
; CHECK-NOT: @f
; CHECK-NOT: @llvm.dbg.declare
; CHECK: %{{[0-9]+}} = load i32, ptr %x.addr.i, align 4, !dbg ![[VAR1:[0-9]+]]
  %call = call i32 @f(i32 3), !dbg !25

; Another test for inlining debug intrinsics where the intrinsic appears at the
; start of the basic block.
; CHECK-NOT: @g
; CHECK-NOT: @llvm.dbg.label
; CHECK: %{{[0-9]+}} = load i32, ptr %x.addr.i1, align 4, !dbg ![[VAR2:[0-9]+]]
  %call1 = call i32 @g(i32 340), !dbg !26
  ret i32 0, !dbg !27
}

; CHECK: ![[VAR1]] = !DILocation(line: 10, scope: ![[SCOPE:[0-9]+]])
; CHECK: ![[VAR2]] = !DILocation(line: 11, scope: ![[SCOPE]])

attributes #0 = { alwaysinline nounwind "no-inline-line-tables" }
attributes #2 = { noinline nounwind optnone "no-inline-line-tables"}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project.git 03ec3a12a94bbbaa11999b6da3a43221a5aa54a5)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "/usr/local/google/home/akhuang/testing/inline-line-tables", checksumkind: CSK_MD5, checksum: "38a4785b48742d3ea655b8f3461436a4")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git 03ec3a12a94bbbaa11999b6da3a43221a5aa54a5)"}
!7 = distinct !DISubprogram(name: "f", scope: !8, file: !8, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DIFile(filename: "t.c", directory: "/usr/local/google/home/akhuang/testing/inline-line-tables", checksumkind: CSK_MD5, checksum: "38a4785b48742d3ea655b8f3461436a4")
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocalVariable(name: "x", arg: 1, scope: !7, file: !8, line: 1, type: !11)
!13 = !DILocation(line: 1, scope: !7)
!14 = !DILocation(line: 2, scope: !7)
!15 = distinct !DISubprogram(name: "g", scope: !8, file: !8, line: 4, type: !9, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!16 = !DILocalVariable(name: "x", arg: 1, scope: !15, file: !8, line: 4, type: !11)
!17 = !DILocation(line: 4, scope: !15)
!18 = !DILabel(scope: !15, name: "L", file: !8, line: 5)
!19 = !DILocation(line: 5, scope: !15)
!20 = !DILocation(line: 6, scope: !15)
!21 = !DILocation(line: 7, scope: !15)
!22 = distinct !DISubprogram(name: "main", scope: !8, file: !8, line: 9, type: !23, scopeLine: 9, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!23 = !DISubroutineType(types: !24)
!24 = !{!11}
!25 = !DILocation(line: 10, scope: !22)
!26 = !DILocation(line: 11, scope: !22)
!27 = !DILocation(line: 12, scope: !22)
