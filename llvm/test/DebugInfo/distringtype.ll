; RUN: llvm-link %s %p/Inputs/distringtype1.ll -S -o - | FileCheck %s

; This test checks that DIStringType is properly linked.

; CHECK: !DIStringType(name: "character(*)!1", size: 32)
; CHECK: !DIStringType(name: "character(*)!2", size: 32)
; CHECK: !DIStringType(name: "character(*)!3", stringLength: !{{[0-9]+}}, stringLengthExpression: !DIExpression(), size: 32)
; CHECK: !DIStringType(name: "character(*)!4", stringLength: !{{[0-9]+}}, stringLengthExpression: !DIExpression(), size: 32)
; CHECK: !DIStringType(name: "character(*)!2", stringLength: !{{[0-9]+}}, stringLengthExpression: !DIExpression(), size: 32)

define void @sub1_(ptr %string1, ptr %string2, i64 %.U0002.arg, i64 %.U0003.arg) !dbg !5 {
L.entry:
  call void @llvm.dbg.declare(metadata ptr %string1, metadata !13, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata ptr %string2, metadata !16, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i64 %.U0002.arg, metadata !11, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i64 %.U0003.arg, metadata !15, metadata !DIExpression()), !dbg !18
  ret void, !dbg !19
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, flags: "'+flang -g distringtype.f90 -S -emit-llvm'", runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4, nameTableKind: None)
!3 = !DIFile(filename: "distringtype.f90", directory: "/tmp/")
!4 = !{}
!5 = distinct !DISubprogram(name: "sub1", scope: !2, file: !3, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !10)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8, !9}
!8 = !DIStringType(name: "character(*)!1", size: 32)
!9 = !DIStringType(name: "character(*)!2", size: 32)
!10 = !{!11, !13, !15, !16}
!11 = !DILocalVariable(arg: 3, scope: !5, file: !3, line: 1, type: !12, flags: DIFlagArtificial)
!12 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "string1", arg: 1, scope: !5, file: !3, line: 1, type: !14)
!14 = !DIStringType(name: "character(*)!3", stringLength: !11, stringLengthExpression: !DIExpression(), size: 32)
!15 = !DILocalVariable(arg: 4, scope: !5, file: !3, line: 1, type: !12, flags: DIFlagArtificial)
!16 = !DILocalVariable(name: "string2", arg: 2, scope: !5, file: !3, line: 1, type: !17)
!17 = !DIStringType(name: "character(*)!4", stringLength: !15, stringLengthExpression: !DIExpression(), size: 32)
!18 = !DILocation(line: 0, scope: !5)
!19 = !DILocation(line: 4, column: 1, scope: !5)
