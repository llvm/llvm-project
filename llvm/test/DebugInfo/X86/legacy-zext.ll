; RUN: llc -filetype=obj -o - %s | llvm-dwarfdump - | FileCheck %s

; CHECK: DW_AT_location (DW_OP_breg5 RDI+0, DW_OP_piece 0x8, DW_OP_lit0, DW_OP_bit_piece 0x40 0x40)

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @t(i64 %x) !dbg !6 {
  call void @llvm.dbg.value(metadata i64 %x, metadata !9,
                            metadata !DIExpression(DW_OP_LLVM_convert, 64, DW_ATE_unsigned,
                                                   DW_OP_LLVM_convert, 128, DW_ATE_unsigned)), !dbg !11
  ret void, !dbg !11
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "legacy-zext.ll", directory: "/")
!2 = !{}
!3 = !{i64 2}
!4 = !{i64 1}
!5 = !{i64 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "t", linkageName: "t", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!7 = !DISubroutineType(types: !2)
!8 = !{!9}
!9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
!10 = !DIBasicType(name: "ty128", size: 128, encoding: DW_ATE_unsigned)
!11 = !DILocation(line: 1, column: 1, scope: !6)
