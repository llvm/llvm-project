; RUN: opt -S -passes=instcombine %s -o - | FileCheck %s

; For performance reasons, we currently limit the number of values that can be
; referenced by a dbg.value to 16. This test checks that we do not exceed this
; limit during salvaging.

; CHECK: DIArgList(i32 poison
; CHECK-NOT: DW_OP_LLVM_arg, 16

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @_Z3foov() local_unnamed_addr !dbg !9 {
entry:
  %call = call i32 @_Z3barv(), !dbg !14
  %call1 = call i32 @_Z3barv(), !dbg !14
  %add = add nsw i32 %call, %call1, !dbg !14
  %call2 = call i32 @_Z3barv(), !dbg !14
  %call4 = call i32 @_Z3barv(), !dbg !14
  %call6 = call i32 @_Z3barv(), !dbg !14
  %call8 = call i32 @_Z3barv(), !dbg !14
  %call10 = call i32 @_Z3barv(), !dbg !14
  %call12 = call i32 @_Z3barv(), !dbg !14
  %call14 = call i32 @_Z3barv(), !dbg !14
  %call16 = call i32 @_Z3barv(), !dbg !14
  %call18 = call i32 @_Z3barv(), !dbg !14
  %call20 = call i32 @_Z3barv(), !dbg !14
  %call22 = call i32 @_Z3barv(), !dbg !14
  %call24 = call i32 @_Z3barv(), !dbg !14
  %call26 = call i32 @_Z3barv(), !dbg !14
  %call28 = call i32 @_Z3barv(), !dbg !14
  %call30 = call i32 @_Z3barv(), !dbg !14
  call void @llvm.dbg.value(metadata !DIArgList(i32 %add, i32 %call30, i32 %call28, i32 %call26, i32 %call24, i32 %call22, i32 %call20, i32 %call18, i32 %call16, i32 %call14, i32 %call12, i32 %call10, i32 %call8, i32 %call6, i32 %call4, i32 %call2), metadata !15, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 15, DW_OP_plus, DW_OP_LLVM_arg, 14, DW_OP_plus, DW_OP_LLVM_arg, 13, DW_OP_plus, DW_OP_LLVM_arg, 12, DW_OP_plus, DW_OP_LLVM_arg, 11, DW_OP_plus, DW_OP_LLVM_arg, 10, DW_OP_plus, DW_OP_LLVM_arg, 9, DW_OP_plus, DW_OP_LLVM_arg, 8, DW_OP_plus, DW_OP_LLVM_arg, 7, DW_OP_plus, DW_OP_LLVM_arg, 6, DW_OP_plus, DW_OP_LLVM_arg, 5, DW_OP_plus, DW_OP_LLVM_arg, 4, DW_OP_plus, DW_OP_LLVM_arg, 3, DW_OP_plus, DW_OP_LLVM_arg, 2, DW_OP_plus, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value)), !dbg !16
  %call32 = call i32 @_Z3barv(), !dbg !17
  ret i32 %call32, !dbg !17
}

declare dso_local i32 @_Z3barv() local_unnamed_addr

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 13.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "limit-arglist-size.cpp", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"uwtable", i32 1}
!7 = !{i32 7, !"frame-pointer", i32 2}
!8 = !{!"clang version 13.0.0"}
!9 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !10, file: !10, line: 3, type: !11, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DIFile(filename: "./limit-arglist-size.cpp", directory: "/")
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DILocation(line: 4, scope: !9)
!15 = !DILocalVariable(name: "v16", scope: !9, file: !10, line: 4, type: !13)
!16 = !DILocation(line: 0, scope: !9)
!17 = !DILocation(line: 5, scope: !9)
