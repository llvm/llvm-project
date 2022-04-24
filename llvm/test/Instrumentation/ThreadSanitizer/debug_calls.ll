; RUN: opt < %s -passes=tsan -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @Increment(i32* nocapture %0) local_unnamed_addr sanitize_thread !dbg !7 {
  call void @llvm.dbg.value(metadata i32* %0, metadata !14, metadata !DIExpression()), !dbg !16
  %2 = load i32, i32* %0, align 4, !dbg !17, !tbaa !18
  call void @llvm.dbg.value(metadata i32 %2, metadata !15, metadata !DIExpression()), !dbg !16
  %3 = add nsw i32 %2, 1, !dbg !22
  store i32 %3, i32* %0, align 4, !dbg !23, !tbaa !18
  ret void, !dbg !24
}
; CHECK-LABEL: define void @Increment
; CHECK-NOT: __tsan_read4
; CHECK: __tsan_write4
; CHECK: ret void

define i32 @NoAccesses(i32 %0) local_unnamed_addr sanitize_thread !dbg !25 {
  call void @llvm.dbg.value(metadata i32 %0, metadata !29, metadata !DIExpression()), !dbg !30
  %2 = add nsw i32 %0, 1, !dbg !31
  ret i32 %2, !dbg !32
}
; CHECK-LABEL: define i32 @NoAccesses
; CHECK-NOT: __tsan_func_entry
; CHECK-NOT: __tsan_func_exit
; CHECK: ret i32

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{!"clang"}
!7 = distinct !DISubprogram(name: "Increment", scope: !8, file: !8, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!8 = !DIFile(filename: "test.c", directory: "")
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{!14, !15}
!14 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !8, line: 1, type: !11)
!15 = !DILocalVariable(name: "x", scope: !7, file: !8, line: 2, type: !12)
!16 = !DILocation(line: 0, scope: !7)
!17 = !DILocation(line: 2, column: 11, scope: !7)
!18 = !{!19, !19, i64 0}
!19 = !{!"int", !20, i64 0}
!20 = !{!"omnipotent char", !21, i64 0}
!21 = !{!"Simple C/C++ TBAA"}
!22 = !DILocation(line: 3, column: 10, scope: !7)
!23 = !DILocation(line: 3, column: 6, scope: !7)
!24 = !DILocation(line: 4, column: 1, scope: !7)
!25 = distinct !DISubprogram(name: "NoAccesses", scope: !8, file: !8, line: 6, type: !26, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !28)
!26 = !DISubroutineType(types: !27)
!27 = !{!12, !12}
!28 = !{!29}
!29 = !DILocalVariable(name: "a", arg: 1, scope: !25, file: !8, line: 6, type: !12)
!30 = !DILocation(line: 0, scope: !25)
!31 = !DILocation(line: 7, column: 12, scope: !25)
!32 = !DILocation(line: 7, column: 3, scope: !25)
