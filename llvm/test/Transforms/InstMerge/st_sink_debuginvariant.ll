; RUN: opt < %s -S -mldst-motion -o - | FileCheck %s
; RUN: opt < %s -S -strip-debug -mldst-motion -o - | FileCheck %s

; Verify that the amount of stores that are sunk is invariant regarding debug
; info.  This used to fail due to including dbg.value instructions when
; calculating the size of a basic block for the "MagicCompileTimeControl"
; check in MergedLoadStoreMotion::mergeStores.

; CHECK-LABEL: return:
; CHECK-NEXT:    %.sink = phi i16 [ 5, %if.end ], [ 6, %if.then ]
; CHECK-NEXT:    %0 = getelementptr inbounds %struct.S0, %struct.S0* %agg.result, i16 0, i32 0
; CHECK-NEXT:    store i16 %.sink, i16* %0
; CHECK-NEXT:    %1 = getelementptr inbounds %struct.S0, %struct.S0* %agg.result, i16 0, i32 1
; CHECK-NEXT:    store i16 0, i16* %1
; CHECK-NEXT:    ret void

%struct.S0 = type { i16, i16 }

@g_173 = dso_local local_unnamed_addr global i16 0, !dbg !0

; Function Attrs: noinline norecurse nounwind
define dso_local void @func_34(%struct.S0* noalias sret %agg.result) local_unnamed_addr #0 !dbg !11 {
entry:
  br i1 undef, label %if.end, label %if.then, !dbg !18

if.then:                                          ; preds = %entry
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i16 5, metadata !19, metadata !DIExpression()), !dbg !22
  %l_303.sroa.0.0..sroa_idx = getelementptr inbounds %struct.S0, %struct.S0* %agg.result, i16 0, i32 0, !dbg !23
  store i16 6, i16* %l_303.sroa.0.0..sroa_idx, !dbg !23
  %l_303.sroa.2.0..sroa_idx1 = getelementptr inbounds %struct.S0, %struct.S0* %agg.result, i16 0, i32 1, !dbg !23
  store i16 0, i16* %l_303.sroa.2.0..sroa_idx1, !dbg !23
  br label %return, !dbg !24

if.end:                                           ; preds = %entry
  %f037 = getelementptr inbounds %struct.S0, %struct.S0* %agg.result, i16 0, i32 0, !dbg !25
  store i16 5, i16* %f037, !dbg !25
  %f138 = getelementptr inbounds %struct.S0, %struct.S0* %agg.result, i16 0, i32 1, !dbg !25
  store i16 0, i16* %f138, !dbg !25
  store i16 0, i16* @g_173, !dbg !27
  br label %return, !dbg !28

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !29
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { noinline norecurse nounwind }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g_173", scope: !2, file: !3, line: 7, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (x)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "csmith11794002068761.c", directory: "")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 1}
!10 = !{!"clang version 7.0.0 (x)"}
!11 = distinct !DISubprogram(name: "func_34", scope: !3, file: !3, line: 16, type: !12, isLocal: false, isDefinition: true, scopeLine: 16, isOptimized: false, unit: !2, retainedNodes: !4)
!12 = !DISubroutineType(types: !13)
!13 = !{!14}
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S0", file: !3, line: 2, size: 32, elements: !15)
!15 = !{!16, !17}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "f0", scope: !14, file: !3, line: 3, baseType: !6, size: 16)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "f1", scope: !14, file: !3, line: 4, baseType: !6, size: 16, offset: 16)
!18 = !DILocation(line: 18, column: 7, scope: !11)
!19 = !DILocalVariable(name: "left", scope: !20, file: !3, line: 23, type: !6)
!20 = distinct !DILexicalBlock(scope: !21, file: !3, line: 18, column: 14)
!21 = distinct !DILexicalBlock(scope: !11, file: !3, line: 18, column: 7)
!22 = !DILocation(line: 23, column: 9, scope: !20)
!23 = !DILocation(line: 26, column: 12, scope: !20)
!24 = !DILocation(line: 26, column: 5, scope: !20)
!25 = !DILocation(line: 29, column: 23, scope: !26)
!26 = distinct !DILexicalBlock(scope: !11, file: !3, line: 28, column: 3)
!27 = !DILocation(line: 30, column: 11, scope: !26)
!28 = !DILocation(line: 31, column: 5, scope: !26)
!29 = !DILocation(line: 33, column: 1, scope: !11)
