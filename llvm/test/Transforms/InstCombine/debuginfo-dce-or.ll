; RUN: opt -instcombine %s -o - -S | FileCheck %s

source_filename = "test.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux"

define void @salvage_or(i64 %context) !dbg !9 {
entry:
  %0 = or i64 %context, 256, !dbg !18
; CHECK:  call void @llvm.dbg.value(metadata i64 %context,
; CHECK-SAME:                       metadata !DIExpression(DW_OP_constu, 256, DW_OP_or, DW_OP_stack_value))
  call void @llvm.dbg.value(metadata i64 %0, metadata !14, metadata !16), !dbg !19
  ret void, !dbg !20
}

declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 7.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{}
!4 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"PIC Level", i32 2}
!8 = !{!"clang version 7.0.0"}
!9 = distinct !DISubprogram(name: "salvage_or", scope: !1, file: !1, line: 91, type: !10, isLocal: false, isDefinition: true, scopeLine: 93, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !12)
!10 = !DISubroutineType(types: !11)
!11 = !{!4}
!12 = !{!13, !14}
!13 = !DILocalVariable(name: "context", arg: 2, scope: !9, file: !1, line: 92, type: !4)
!14 = !DILocalVariable(name: "master", scope: !9, file: !1, line: 94, type: !4)
!15 = !{}
!16 = !DIExpression()
!17 = !DILexicalBlockFile(scope: !9, file: !1, discriminator: 2)
!18 = !DILocation(line: 94, column: 18, scope: !17)
!19 = !DILocation(line: 94, column: 18, scope: !9)
!20 = !DILocation(line: 126, column: 1, scope: !9)
