; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; Check for some sane output (original problem was a crash).
; CHECK: DEBUG_VALUE: fred:Count <- 0

target triple = "hexagon"

; Function Attrs: nounwind
define i32 @f0(i32 %a0, ptr %a1) local_unnamed_addr #0 !dbg !5 {
b0:
  br label %b1

b1:                                               ; preds = %b0
  call void @llvm.dbg.value(metadata i32 0, metadata !8, metadata !DIExpression()), !dbg !10
  br label %b2

b2:                                               ; preds = %b3, %b1
  %v0 = phi i32 [ 0, %b1 ], [ %v2, %b3 ]
  %v1 = tail call ptr %a1(i32 12, ptr null) #0
  br label %b3

b3:                                               ; preds = %b2
  %v2 = add nuw i32 %v0, 1
  %v3 = icmp ult i32 %v2, %a0
  br i1 %v3, label %b2, label %b4

b4:                                               ; preds = %b3
  ret i32 0
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "file.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "fred", scope: !1, file: !1, line: 116, type: !6, isLocal: false, isDefinition: true, scopeLine: 121, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !7)
!6 = !DISubroutineType(types: !2)
!7 = !{!8}
!8 = !DILocalVariable(name: "Count", scope: !5, file: !1, line: 1, type: !9)
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 1, column: 1, scope: !5, inlinedAt: !11)
!11 = distinct !DILocation(line: 1, column: 1, scope: !5)
