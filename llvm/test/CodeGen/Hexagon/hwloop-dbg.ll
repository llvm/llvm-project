; RUN: llc < %s -mtriple=hexagon -disable-lsr | FileCheck %s

; CHECK:     loop0(
; CHECK-NOT: add({{r[0-9]*}}, #
; CHECK:     endloop0

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

define void @f0(ptr nocapture %a0, ptr nocapture %a1) #0 !dbg !4 {
b0:
  call void @llvm.dbg.value(metadata ptr %a0, metadata !10, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.value(metadata ptr %a1, metadata !11, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !16
  br label %b1, !dbg !16

b1:                                               ; preds = %b1, %b0
  %v0 = phi ptr [ %a0, %b0 ], [ %v7, %b1 ]
  %v1 = phi i32 [ 0, %b0 ], [ %v5, %b1 ]
  %v2 = phi ptr [ %a1, %b0 ], [ %v3, %b1 ]
  %v3 = getelementptr inbounds i32, ptr %v2, i32 1, !dbg !18
  call void @llvm.dbg.value(metadata ptr %v3, metadata !11, metadata !DIExpression()), !dbg !18
  %v4 = load i32, ptr %v2, align 4, !dbg !18
  store i32 %v4, ptr %v0, align 4, !dbg !18
  %v5 = add nsw i32 %v1, 1, !dbg !20
  call void @llvm.dbg.value(metadata i32 %v5, metadata !12, metadata !DIExpression()), !dbg !20
  %v6 = icmp eq i32 %v5, 10, !dbg !16
  %v7 = getelementptr i32, ptr %v0, i32 1
  br i1 %v6, label %b2, label %b1, !dbg !16

b2:                                               ; preds = %b1
  ret void, !dbg !21
}

declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "QuIC LLVM Hexagon Clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2)
!1 = !DIFile(filename: "hwloop-dbg.c", directory: "/test")
!2 = !{}
!3 = !{i32 1, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", scope: null, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !9)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7, !7}
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 32, align: 32)
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{!10, !11, !12}
!10 = !DILocalVariable(name: "a", arg: 1, scope: !4, file: !1, line: 1, type: !7)
!11 = !DILocalVariable(name: "b", arg: 2, scope: !4, file: !1, line: 1, type: !7)
!12 = !DILocalVariable(name: "i", scope: !13, file: !1, line: 2, type: !8)
!13 = distinct !DILexicalBlock(scope: !4, file: !1, line: 1, column: 26)
!14 = !DILocation(line: 1, column: 15, scope: !4)
!15 = !DILocation(line: 1, column: 23, scope: !4)
!16 = !DILocation(line: 3, column: 8, scope: !17)
!17 = distinct !DILexicalBlock(scope: !13, file: !1, line: 3, column: 3)
!18 = !DILocation(line: 4, column: 5, scope: !19)
!19 = distinct !DILexicalBlock(scope: !17, file: !1, line: 3, column: 28)
!20 = !DILocation(line: 3, column: 23, scope: !17)
!21 = !DILocation(line: 6, column: 1, scope: !13)
