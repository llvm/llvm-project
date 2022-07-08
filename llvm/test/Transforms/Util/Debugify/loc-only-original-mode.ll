; RUN: opt < %s -deadargelim \
; RUN:     -verify-each-debuginfo-preserve \
; RUN:     -debugify-level=locations -S 2>&1 | FileCheck %s

; RUN: opt < %s -deadargelim \
; RUN:     -verify-each-debuginfo-preserve \
; RUN:     -debugify-level=location+variables -S 2>&1 | FileCheck %s --check-prefix=CHECK-DROP

; RUN: opt < %s -deadargelim \
; RUN:     -verify-each-debuginfo-preserve \
; RUN:     -debugify-func-limit=0 -S 2>&1 | FileCheck %s

; RUN: opt < %s -deadargelim \
; RUN:     -verify-each-debuginfo-preserve \
; RUN:     -debugify-func-limit=2 -S 2>&1 | FileCheck %s --check-prefix=CHECK-DROP


; CHECK-NOT: drops dbg.value()/dbg.declare()
; CHECK-DROP: drops dbg.value()/dbg.declare()

target triple = "x86_64-unknown-linux-gnu"

define dso_local i32 @fn2(i32 %l, i32 %k) !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 %l, metadata !12, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.value(metadata i32 %k, metadata !13, metadata !DIExpression()), !dbg !15
  %call = call i32 (...) @fn3(), !dbg !16
  call void @llvm.dbg.value(metadata i32 %call, metadata !14, metadata !DIExpression()), !dbg !15
  ret i32 %call, !dbg !17
}

declare !dbg !18 dso_local i32 @fn3(...)

define dso_local i32 @fn(i32 %x, i32 %y) !dbg !22 {
entry:
  call void @llvm.dbg.value(metadata i32 %x, metadata !24, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 %y, metadata !25, metadata !DIExpression()), !dbg !27
  %call = call i32 @fn2(i32 %x, i32 %y), !dbg !27
  call void @llvm.dbg.value(metadata i32 %call, metadata !26, metadata !DIExpression()), !dbg !27
  %add = add nsw i32 %call, %x, !dbg !27
  %add1 = add nsw i32 %add, %y, !dbg !27
  ret i32 %add1, !dbg !27
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/dir")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{!"clang version 14.0.0"}
!7 = distinct !DISubprogram(name: "fn2", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0,
 retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12, !13, !14}
!12 = !DILocalVariable(name: "l", arg: 1, scope: !7, file: !1, line: 3, type: !10)
!13 = !DILocalVariable(name: "k", arg: 2, scope: !7, file: !1, line: 3, type: !10)
!14 = !DILocalVariable(name: "s", scope: !7, file: !1, line: 4, type: !10)
!15 = !DILocation(line: 0, scope: !7)
!16 = !DILocation(line: 4, column: 11, scope: !7)
!17 = !DILocation(line: 5, column: 3, scope: !7)
!18 = !DISubprogram(name: "fn3", scope: !1, file: !1, line: 1, type: !19, spFlags: DISPFlagOptimized, retainedNodes: !21)
!19 = !DISubroutineType(types: !20)
!20 = !{!10}
!21 = !{}
!22 = distinct !DISubprogram(name: "fn", scope: !1, file: !1, line: 8, type: !8, scopeLine: 8, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0,
 retainedNodes: !23)
!23 = !{!24, !25, !26}
!24 = !DILocalVariable(name: "x", arg: 1, scope: !22, file: !1, line: 8, type: !10)
!25 = !DILocalVariable(name: "y", arg: 2, scope: !22, file: !1, line: 8, type: !10)
!26 = !DILocalVariable(name: "local", scope: !22, file: !1, line: 9, type: !10)
!27 = !DILocation(line: 0, scope: !22)
