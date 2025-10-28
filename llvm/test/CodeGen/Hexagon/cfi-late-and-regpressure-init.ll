; RUN: llc -mtriple=hexagon -enable-misched=true < %s | FileCheck %s
; This test checks the delayed emission of CFI instructions
; This test also checks the proper initialization of RegisterPressureTracker.
; The RegisterPressureTracker must skip debug instructions upon entry of a BB

target triple = "hexagon-unknown--elf"

; Check that allocframe was packetized with the two adds.
; CHECK: f0:
; CHECK: {
; CHECK-DAG: allocframe
; CHECK-DAG: add
; CHECK-DAG: add
; CHECK: }
; CHECK: dealloc_return
; CHECK: }


; Function Attrs: nounwind
define i32 @f0(i32 %a0, i32 %a1) #0 !dbg !5 {
b0:
  call void @llvm.dbg.value(metadata i32 %a0, metadata !10, metadata !DIExpression()), !dbg !12
  call void @llvm.dbg.value(metadata i32 %a1, metadata !11, metadata !DIExpression()), !dbg !13
  %v0 = add nsw i32 %a0, 1, !dbg !14
  %v1 = add nsw i32 %a1, 1, !dbg !15
  %v2 = tail call i32 @f1(i32 %v0, i32 %v1) #3, !dbg !16
  %v3 = add nsw i32 %v2, 1, !dbg !17
  ret i32 %v3, !dbg !18
}

declare i32 @f1(i32, i32) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { "target-cpu"="hexagonv55" }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/test")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !6, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !9)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8, !8}
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{!10, !11}
!10 = !DILocalVariable(name: "x", arg: 1, scope: !5, file: !1, line: 3, type: !8)
!11 = !DILocalVariable(name: "y", arg: 2, scope: !5, file: !1, line: 3, type: !8)
!12 = !DILocation(line: 3, column: 13, scope: !5)
!13 = !DILocation(line: 3, column: 20, scope: !5)
!14 = !DILocation(line: 4, column: 15, scope: !5)
!15 = !DILocation(line: 4, column: 20, scope: !5)
!16 = !DILocation(line: 4, column: 10, scope: !5)
!17 = !DILocation(line: 4, column: 24, scope: !5)
!18 = !DILocation(line: 4, column: 3, scope: !5)
