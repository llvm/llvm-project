; RUN: opt -verify < %s | opt -verify -S | FileCheck %s

; Simple IR-BC-IR round-trip test for a @llvm.dbg.value that uses !DIArgList
; and DW_OP_LLVM_arg.

source_filename = ".\\debug_value_list.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.16.27034"

; CHECK-COUNT-3: llvm.dbg.value(
; CHECK-SAME: metadata !DIArgList(i32 %a, i32 %b)
; CHECK-SAME: metadata !16,
; CHECK-SAME: metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus)
define dso_local i32 @"?foo@@YAHHH@Z"(i32 %a, i32 %b) local_unnamed_addr !dbg !8 {
entry:
  call void @llvm.dbg.value(metadata !DIArgList(i32 %b), metadata !14, metadata !DIExpression(DW_OP_LLVM_arg, 0)), !dbg !17
  call void @llvm.dbg.value(metadata !DIArgList(i32 %a), metadata !15, metadata !DIExpression(DW_OP_LLVM_arg, 0)), !dbg !17
  call void @llvm.dbg.value(
    metadata !DIArgList(i32 %a, i32 %b),
    metadata !16,
    metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus)), !dbg !17
  %mul = mul nsw i32 %b, %a, !dbg !18
  ret i32 %mul, !dbg !18
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 11.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "debug_value_list.cpp", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 11.0.0"}
!8 = distinct !DISubprogram(name: "foo", linkageName: "?foo@@YAHHH@Z", scope: !9, file: !9, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!9 = !DIFile(filename: ".\\debug_value_list.cpp", directory: "/tmp")
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12, !12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{!14, !15, !16}
!14 = !DILocalVariable(name: "b", arg: 2, scope: !8, file: !9, line: 1, type: !12)
!15 = !DILocalVariable(name: "a", arg: 1, scope: !8, file: !9, line: 1, type: !12)
!16 = !DILocalVariable(name: "c", scope: !8, file: !9, line: 2, type: !12)
!17 = !DILocation(line: 0, scope: !8)
!18 = !DILocation(line: 3, scope: !8)
