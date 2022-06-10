; RUN: opt -S %s -early-cse -earlycse-debug-hash | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

@a = global i8 25, align 1, !dbg !0

define signext i16 @b() !dbg !12 {
entry:
  call void @llvm.dbg.value(metadata i16 23680, metadata !17, metadata !DIExpression()), !dbg !18
  %0 = load i8, ptr @a, align 1, !dbg !19, !tbaa !20
  %conv = sext i8 %0 to i16, !dbg !19

; CHECK: call void @llvm.dbg.value(metadata i8 %0, metadata !17, metadata !DIExpression(DW_OP_LLVM_convert, 8, DW_ATE_signed, DW_OP_LLVM_convert, 16, DW_ATE_signed, DW_OP_stack_value)), !dbg !18
; CHECK-NEXT:  call i32 (...) @optimize_me_not()

  call void @llvm.dbg.value(metadata i16 %conv, metadata !17, metadata !DIExpression()), !dbg !18
  %call = call i32 (...) @optimize_me_not(), !dbg !23
  %1 = load i8, ptr @a, align 1, !dbg !24, !tbaa !20
  %conv1 = sext i8 %1 to i16, !dbg !24
  ret i16 %conv1, !dbg !25
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare i32 @optimize_me_not(...)

define i32 @main() !dbg !26 {
entry:
  %call = call signext i16 @b(), !dbg !30
  ret i32 0, !dbg !31
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 8.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: GNU)
!3 = !DIFile(filename: "patatino.c", directory: "/Users/davide/llvm-monorepo/llvm-mono/build/bin")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 7, !"PIC Level", i32 2}
!11 = !{!"clang version 8.0.0 "}
!12 = distinct !DISubprogram(name: "b", scope: !3, file: !3, line: 2, type: !13, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !16)
!13 = !DISubroutineType(types: !14)
!14 = !{!15}
!15 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!16 = !{!17}
!17 = !DILocalVariable(name: "i", scope: !12, file: !3, line: 3, type: !15)
!18 = !DILocation(line: 3, column: 9, scope: !12)
!19 = !DILocation(line: 4, column: 7, scope: !12)
!20 = !{!21, !21, i64 0}
!21 = !{!"omnipotent char", !22, i64 0}
!22 = !{!"Simple C/C++ TBAA"}
!23 = !DILocation(line: 5, column: 3, scope: !12)
!24 = !DILocation(line: 6, column: 10, scope: !12)
!25 = !DILocation(line: 6, column: 3, scope: !12)
!26 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 8, type: !27, scopeLine: 8, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!27 = !DISubroutineType(types: !28)
!28 = !{!29}
!29 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!30 = !DILocation(line: 8, column: 14, scope: !26)
!31 = !DILocation(line: 8, column: 19, scope: !26)
