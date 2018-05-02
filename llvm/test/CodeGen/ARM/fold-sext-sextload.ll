; RUN: llc -mtriple armv7 %s -stop-before=livedebugvalues -o - | FileCheck %s

define <4 x i8> @i(<4 x i8>*) !dbg !8 {
  %2 = load <4 x i8>, <4 x i8>* %0, align 4, !dbg !14
  ; CHECK: $[[reg:.*]] = VLD1LNd32 {{.*}} debug-location !14 :: (load 4 from %ir.0)
  ; CHECK-NEXT: VMOVLsv8i16 {{.*}} $[[reg]], {{.*}} debug-location !14
  ; CHECK-NEXT: VMOVLsv4i32 {{.*}} $[[reg]], {{.*}} debug-location !14

  %3 = sdiv <4 x i8> zeroinitializer, %2, !dbg !15
  call void @llvm.dbg.value(metadata <4 x i8> %2, metadata !11, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.value(metadata <4 x i8> %3, metadata !13, metadata !DIExpression()), !dbg !15
  ret <4 x i8> %3, !dbg !16
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.debugify = !{!0, !1, !2, !3}
!llvm.module.flags = !{!4}
!llvm.dbg.cu = !{!5}

!0 = !{i32 24}
!1 = !{i32 19}
!2 = !{i32 3}
!3 = !{i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !7)
!6 = !DIFile(filename: "/Users/vsk/Desktop/test.ll", directory: "/")
!7 = !{}
!8 = distinct !DISubprogram(name: "i", linkageName: "i", scope: null, file: !6, line: 1, type: !9, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !5, variables: !10)
!9 = !DISubroutineType(types: !7)
!10 = !{!11, !13}
!11 = !DILocalVariable(name: "1", scope: !8, file: !6, line: 1, type: !12)
!12 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!13 = !DILocalVariable(name: "2", scope: !8, file: !6, line: 2, type: !12)
!14 = !DILocation(line: 1, column: 1, scope: !8)
!15 = !DILocation(line: 2, column: 1, scope: !8)
!16 = !DILocation(line: 3, column: 1, scope: !8)
!17 = !{i32 2, !"Debug Info Version", i32 3}
