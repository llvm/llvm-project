; RUN: llc -O2 < %s
; REQUIRES: asserts

target triple = "hexagon"

%struct.wombat.8.56.133.143.153.163.173.183.212.232.281.310.358.406.548.656.666.676.686.696.751.760.850.859 = type { ptr, ptr, ptr, ptr, ptr, ptr, i8, i8, ptr, i32, i32, ptr, %struct.wombat.1.7.55.132.142.152.162.172.182.211.231.280.309.357.405.547.655.665.675.685.695.750.759.849.858 }
%struct.wombat.1.7.55.132.142.152.162.172.182.211.231.280.309.357.405.547.655.665.675.685.695.750.759.849.858 = type { ptr, i32, i32, i32, i32, i32, i32, i32 }

@global = external dso_local local_unnamed_addr global %struct.wombat.8.56.133.143.153.163.173.183.212.232.281.310.358.406.548.656.666.676.686.696.751.760.850.859, align 4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.label(metadata) #0

define dso_local i32 @pluto(i8 %arg) local_unnamed_addr #1 !dbg !164 !prof !173 {
bb:
  %cond0 = icmp ne i8 %arg, 0, !dbg !174
  br i1 %cond0, label %bb6, label %bb65, !dbg !174

bb6:                                              ; preds = %bb
  %tmp7 = load i8, ptr getelementptr inbounds (%struct.wombat.8.56.133.143.153.163.173.183.212.232.281.310.358.406.548.656.666.676.686.696.751.760.850.859, ptr @global, i32 0, i32 6), align 4, !dbg !175
  switch i8 %tmp7, label %bb65 [
    i8 0, label %bb24
    i8 2, label %bb24
  ], !dbg !176

bb24:                                             ; preds = %bb6, %bb6
  %cond1 = icmp eq i8 %arg, 2, !dbg !184
  br i1 %cond1, label %bb27, label %bb65, !dbg !184

bb27:                                             ; preds = %bb24
  switch i8 %arg, label %bb65 [
    i8 0, label %bb28
    i8 2, label %bb45
  ], !dbg !185

bb28:                                             ; preds = %bb27
  %tmp35 = tail call i32 @wombat(i32 3, ptr null, i8 zeroext 0) #2, !dbg !211
  br label %bb62, !dbg !212

bb45:                                             ; preds = %bb27
  br label %bb62, !dbg !214

bb62:                                             ; preds = %bb45, %bb28
  %tmp64 = icmp eq i32 0, 0, !dbg !222
  br i1 %tmp64, label %bb68, label %bb65

bb65:                                             ; preds = %bb62, %bb27, %bb24, %bb6, %bb
  %tmp66 = phi i32 [ 1, %bb24 ], [ 0, %bb ], [ 1, %bb62 ], [ 3, %bb6 ], [ 3, %bb27 ]
  %tmp67 = phi i8 [ %tmp7, %bb24 ], [ 0, %bb ], [ %tmp7, %bb62 ], [ %tmp7, %bb6 ], [ %tmp7, %bb27 ]
  call void @llvm.dbg.label(metadata !172), !dbg !223
  store i8 %tmp67, ptr getelementptr inbounds (%struct.wombat.8.56.133.143.153.163.173.183.212.232.281.310.358.406.548.656.666.676.686.696.751.760.850.859, ptr @global, i32 0, i32 6), align 4, !dbg !224
  br label %bb68, !dbg !227

bb68:                                             ; preds = %bb65, %bb62
  %tmp69 = phi i32 [ %tmp66, %bb65 ], [ 0, %bb62 ]
  ret i32 %tmp69, !dbg !228
}

declare dso_local void @barney(ptr) local_unnamed_addr #1

declare dso_local i32 @eggs() local_unnamed_addr #1

declare dso_local i32 @wombat(i32, ptr, i8) local_unnamed_addr #1

declare dso_local void @barney.1(ptr) local_unnamed_addr #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!136, !137}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "foo.c", directory: "/tmp")
!2 = !{}
!3 = !{!4, !6}
!4 = !DIDerivedType(tag: DW_TAG_typedef, name: "tempName", file: !1, line: 38, baseType: !5)
!5 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 32)
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "tempName", file: !1, line: 44, baseType: !8)
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "tempName", file: !1, line: 45, size: 96, elements: !9)
!9 = !{!10, !13, !135}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !8, file: !1, line: 47, baseType: !11, size: 32)
!11 = !DIDerivedType(tag: DW_TAG_typedef, name: "tempName", file: !1, line: 28, baseType: !12)
!12 = !DIBasicType(name: "long unsigned int", size: 32, encoding: DW_ATE_unsigned)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !8, file: !1, line: 48, baseType: !14, size: 32, offset: 32)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 32)
!15 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !16)
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "tempName", file: !1, line: 26, baseType: !17)
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "tempName", file: !1, line: 27, size: 640, elements: !18)
!18 = !{!19, !97, !102, !106, !110, !114, !115, !116, !122, !126, !130, !131}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !17, file: !1, line: 29, baseType: !20, size: 288)
!20 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "tempName", file: !1, line: 85, size: 288, elements: !21)
!21 = !{!22, !55, !59, !63, !64, !68, !69, !83, !89}
!22 = !DIDerivedType(tag: DW_TAG_member, scope: !20, file: !1, line: 87, baseType: !23, size: 32)
!23 = distinct !DICompositeType(tag: DW_TAG_union_type, scope: !20, file: !1, line: 87, size: 32, elements: !24)
!24 = !{!25, !51}
!25 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !23, file: !1, line: 89, baseType: !26, size: 32)
!26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !27, size: 32)
!27 = !DISubroutineType(types: !28)
!28 = !{!29, !31, !34, !35}
!29 = !DIDerivedType(tag: DW_TAG_typedef, name: "tempName", file: !1, line: 82, baseType: !30)
!30 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !32, size: 32)
!32 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !33)
!33 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_unsigned_char)
!34 = !DIDerivedType(tag: DW_TAG_typedef, name: "tempName", file: !1, line: 77, baseType: !11)
!35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !36, size: 32)
!36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !37, size: 32)
!37 = !DIDerivedType(tag: DW_TAG_typedef, name: "tempName", file: !1, line: 83, baseType: !38)
!38 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "tempName", file: !1, line: 110, size: 128, elements: !39)
!39 = !{!40, !41, !48, !50}
!40 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !38, file: !1, line: 112, baseType: !11, size: 32)
!41 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !38, file: !1, line: 113, baseType: !42, size: 32, offset: 32)
!42 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !43, size: 32)
!43 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !44)
!44 = !DIDerivedType(tag: DW_TAG_typedef, name: "tempName", file: !1, line: 103, baseType: !45)
!45 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "tempName", file: !1, line: 104, size: 288, elements: !46)
!46 = !{!47}
!47 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !45, file: !1, line: 106, baseType: !20, size: 288)
!48 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !38, file: !1, line: 114, baseType: !49, size: 32, offset: 64)
!49 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 32)
!50 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !38, file: !1, line: 115, baseType: !11, size: 32, offset: 96)
!51 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !23, file: !1, line: 90, baseType: !52, size: 32)
!52 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !53, size: 32)
!53 = !DISubroutineType(types: !54)
!54 = !{!29, !31, !34, !35, !30}
!55 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !20, file: !1, line: 92, baseType: !56, size: 32, offset: 32)
!56 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !57, size: 32)
!57 = !DISubroutineType(types: !58)
!58 = !{!11, !11, !36}
!59 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !20, file: !1, line: 93, baseType: !60, size: 32, offset: 64)
!60 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !61, size: 32)
!61 = !DISubroutineType(types: !62)
!62 = !{!29, !11, !36}
!63 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !20, file: !1, line: 94, baseType: !60, size: 32, offset: 96)
!64 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !20, file: !1, line: 95, baseType: !65, size: 32, offset: 128)
!65 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !66, size: 32)
!66 = !DISubroutineType(types: !67)
!67 = !{!29, !11, !36, !11}
!68 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !20, file: !1, line: 96, baseType: !60, size: 32, offset: 160)
!69 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !20, file: !1, line: 97, baseType: !70, size: 32, offset: 192)
!70 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !71, size: 32)
!71 = !DISubroutineType(types: !72)
!72 = !{!29, !11, !36, !73, !11}
!73 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !74, size: 32)
!74 = !DIDerivedType(tag: DW_TAG_typedef, name: "tempName", file: !1, line: 62, baseType: !75)
!75 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "tempName", file: !1, line: 63, size: 320, elements: !76)
!76 = !{!77, !78, !79}
!77 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !75, file: !1, line: 65, baseType: !11, size: 32)
!78 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !75, file: !1, line: 66, baseType: !11, size: 32, offset: 32)
!79 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !75, file: !1, line: 67, baseType: !80, size: 256, offset: 64)
!80 = !DICompositeType(tag: DW_TAG_array_type, baseType: !33, size: 256, elements: !81)
!81 = !{!82}
!82 = !DISubrange(count: 32)
!83 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !20, file: !1, line: 98, baseType: !84, size: 32, offset: 224)
!84 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !85, size: 32)
!85 = !DISubroutineType(types: !86)
!86 = !{!29, !11, !36, !87, !88}
!87 = !DIDerivedType(tag: DW_TAG_typedef, name: "tempName", file: !1, line: 78, baseType: !11)
!88 = !DIDerivedType(tag: DW_TAG_typedef, name: "tempName", file: !1, line: 79, baseType: !11)
!89 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !20, file: !1, line: 99, baseType: !90, size: 32, offset: 256)
!90 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !91, size: 32)
!91 = !DISubroutineType(types: !92)
!92 = !{!29, !11, !36, !93, !94, !11, !49, !11, !96}
!93 = !DIDerivedType(tag: DW_TAG_typedef, name: "tempName", file: !1, line: 80, baseType: !11)
!94 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !95, size: 32)
!95 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!96 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 32)
!97 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !17, file: !1, line: 30, baseType: !98, size: 32, offset: 288)
!98 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !99, size: 32)
!99 = !DISubroutineType(types: !100)
!100 = !{!29, !11, !36, !101}
!101 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 32)
!102 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !17, file: !1, line: 31, baseType: !103, size: 32, offset: 320)
!103 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !104, size: 32)
!104 = !DISubroutineType(types: !105)
!105 = !{!29, !11, !36, !4}
!106 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !17, file: !1, line: 32, baseType: !107, size: 32, offset: 352)
!107 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !108, size: 32)
!108 = !DISubroutineType(types: !109)
!109 = !{!29, !11, !36, !96, !96}
!110 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !17, file: !1, line: 33, baseType: !111, size: 32, offset: 384)
!111 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !112, size: 32)
!112 = !DISubroutineType(types: !113)
!113 = !{!29, !11, !36, !11, !11}
!114 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !17, file: !1, line: 35, baseType: !60, size: 32, offset: 416)
!115 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !17, file: !1, line: 36, baseType: !60, size: 32, offset: 448)
!116 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !17, file: !1, line: 37, baseType: !117, size: 32, offset: 480)
!117 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !118, size: 32)
!118 = !DISubroutineType(types: !119)
!119 = !{!29, !11, !36, !120, !11, !96}
!120 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !121, size: 32)
!121 = !DIDerivedType(tag: DW_TAG_typedef, name: "tempName", file: !1, line: 72, baseType: !5)
!122 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !17, file: !1, line: 38, baseType: !123, size: 32, offset: 512)
!123 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !124, size: 32)
!124 = !DISubroutineType(types: !125)
!125 = !{!29, !11, !36, !30}
!126 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !17, file: !1, line: 39, baseType: !127, size: 32, offset: 544)
!127 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !128, size: 32)
!128 = !DISubroutineType(types: !129)
!129 = !{!29, !11, !36, !11, !96}
!130 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !17, file: !1, line: 40, baseType: !111, size: 32, offset: 576)
!131 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !17, file: !1, line: 41, baseType: !132, size: 32, offset: 608)
!132 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !133, size: 32)
!133 = !DISubroutineType(types: !134)
!134 = !{!29, !11, !36, !120, !11}
!135 = !DIDerivedType(tag: DW_TAG_member, name: "tempName", scope: !8, file: !1, line: 49, baseType: !49, size: 32, offset: 64)
!136 = !{i32 2, !"Debug Info Version", i32 3}
!137 = !{i32 1, !"ProfileSummary", !138}
!138 = !{!139, !140, !141, !142, !143, !144, !145, !146}
!139 = !{!"ProfileFormat", !"SampleProfile"}
!140 = !{!"TotalCount", i64 12081434}
!141 = !{!"MaxCount", i64 59842}
!142 = !{!"MaxInternalCount", i64 0}
!143 = !{!"MaxFunctionCount", i64 2338166}
!144 = !{!"NumCounts", i64 63284}
!145 = !{!"NumFunctions", i64 6868}
!146 = !{!"DetailedSummary", !147}
!147 = !{!148, !149, !150, !151, !152, !153, !154, !155, !156, !157, !158, !159, !160, !161, !162, !163}
!148 = !{i32 10000, i64 59838, i32 3}
!149 = !{i32 100000, i64 17206, i32 45}
!150 = !{i32 200000, i64 11480, i32 138}
!151 = !{i32 300000, i64 6378, i32 288}
!152 = !{i32 400000, i64 3286, i32 563}
!153 = !{i32 500000, i64 2201, i32 1035}
!154 = !{i32 600000, i64 1285, i32 1800}
!155 = !{i32 700000, i64 726, i32 3144}
!156 = !{i32 800000, i64 421, i32 5243}
!157 = !{i32 900000, i64 246, i32 9701}
!158 = !{i32 950000, i64 239, i32 12082}
!159 = !{i32 990000, i64 32, i32 17252}
!160 = !{i32 999000, i64 4, i32 27541}
!161 = !{i32 999900, i64 1, i32 33470}
!162 = !{i32 999990, i64 1, i32 33470}
!163 = !{i32 999999, i64 1, i32 33470}
!164 = distinct !DISubprogram(name: "qdss_control_set_sink", scope: !1, file: !1, line: 259, type: !165, scopeLine: 260, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !167)
!165 = !DISubroutineType(types: !166)
!166 = !{!30, !4}
!167 = !{!168, !169, !170, !171, !172}
!168 = !DILocalVariable(name: "sinkid", arg: 1, scope: !164, file: !1, line: 259, type: !4)
!169 = !DILocalVariable(name: "nErr", scope: !164, file: !1, line: 261, type: !30)
!170 = !DILocalVariable(name: "new_trace_sink", scope: !164, file: !1, line: 262, type: !4)
!171 = !DILocalVariable(name: "current_trace_sink", scope: !164, file: !1, line: 263, type: !4)
!172 = !DILabel(scope: !164, name: "tempName", file: !1, line: 282)
!173 = !{!"function_entry_count", i64 -1}
!174 = !DILocation(line: 272, column: 8, scope: !164)
!175 = !DILocation(line: 274, column: 30, scope: !164)
!176 = !DILocation(line: 176, column: 4, scope: !177, inlinedAt: !182)
!177 = distinct !DISubprogram(name: "qdss_trace_sink_stop", scope: !1, file: !1, line: 172, type: !165, scopeLine: 173, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !178)
!178 = !{!179, !180, !181}
!179 = !DILocalVariable(name: "trace_sink", arg: 1, scope: !177, file: !1, line: 172, type: !4)
!180 = !DILocalVariable(name: "nErr", scope: !177, file: !1, line: 174, type: !30)
!181 = !DILabel(scope: !177, name: "tempName", file: !1, line: 192)
!182 = distinct !DILocation(line: 276, column: 22, scope: !183)
!183 = distinct !DILexicalBlock(scope: !164, file: !1, line: 276, column: 8)
!184 = !DILocation(line: 276, column: 8, scope: !164)
!185 = !DILocation(line: 140, column: 4, scope: !186, inlinedAt: !191)
!186 = distinct !DISubprogram(name: "qdss_trace_sink_start", scope: !1, file: !1, line: 136, type: !165, scopeLine: 137, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !187)
!187 = !{!188, !189, !190}
!188 = !DILocalVariable(name: "trace_sink", arg: 1, scope: !186, file: !1, line: 136, type: !4)
!189 = !DILocalVariable(name: "nErr", scope: !186, file: !1, line: 138, type: !30)
!190 = !DILabel(scope: !186, name: "tempName", file: !1, line: 156)
!191 = distinct !DILocation(line: 278, column: 22, scope: !192)
!192 = distinct !DILexicalBlock(scope: !164, file: !1, line: 278, column: 8)
!193 = !DILocation(line: 65, column: 40, scope: !194, inlinedAt: !200)
!194 = distinct !DISubprogram(name: "DalTMC_SetMode", scope: !1, file: !1, line: 63, type: !195, scopeLine: 64, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !197)
!195 = !DISubroutineType(types: !196)
!196 = !{!29, !36, !4}
!197 = !{!198, !199}
!198 = !DILocalVariable(name: "_h", arg: 1, scope: !194, file: !1, line: 63, type: !36)
!199 = !DILocalVariable(name: "mode", arg: 2, scope: !194, file: !1, line: 63, type: !4)
!200 = distinct !DILocation(line: 65, column: 22, scope: !201, inlinedAt: !208)
!201 = distinct !DILexicalBlock(scope: !202, file: !1, line: 65, column: 8)
!202 = distinct !DISubprogram(name: "qdss_trace_cbuf_start", scope: !1, file: !1, line: 61, type: !203, scopeLine: 62, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !205)
!203 = !DISubroutineType(types: !204)
!204 = !{!30}
!205 = !{!206, !207}
!206 = !DILocalVariable(name: "nErr", scope: !202, file: !1, line: 63, type: !30)
!207 = !DILabel(scope: !202, name: "tempName", file: !1, line: 68)
!208 = distinct !DILocation(line: 142, column: 16, scope: !209, inlinedAt: !191)
!209 = distinct !DILexicalBlock(scope: !210, file: !1, line: 142, column: 11)
!210 = distinct !DILexicalBlock(scope: !186, file: !1, line: 140, column: 23)
!211 = !DILocation(line: 65, column: 11, scope: !194, inlinedAt: !200)
!212 = !DILocation(line: 0, scope: !213, inlinedAt: !208)
!213 = distinct !DILexicalBlock(scope: !202, file: !1, line: 66, column: 8)
!214 = !DILocation(line: 0, scope: !215, inlinedAt: !220)
!215 = distinct !DILexicalBlock(scope: !216, file: !1, line: 101, column: 8)
!216 = distinct !DISubprogram(name: "qdss_trace_hwfifo_start", scope: !1, file: !1, line: 95, type: !203, scopeLine: 96, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !217)
!217 = !{!218, !219}
!218 = !DILocalVariable(name: "nErr", scope: !216, file: !1, line: 97, type: !30)
!219 = !DILabel(scope: !216, name: "tempName", file: !1, line: 103)
!220 = distinct !DILocation(line: 147, column: 16, scope: !221, inlinedAt: !191)
!221 = distinct !DILexicalBlock(scope: !210, file: !1, line: 147, column: 11)
!222 = !DILocation(line: 0, scope: !210, inlinedAt: !191)
!223 = !DILocation(line: 282, column: 4, scope: !164)
!224 = !DILocation(line: 284, column: 31, scope: !225)
!225 = distinct !DILexicalBlock(scope: !226, file: !1, line: 282, column: 28)
!226 = distinct !DILexicalBlock(scope: !164, file: !1, line: 282, column: 18)
!227 = !DILocation(line: 285, column: 4, scope: !225)
!228 = !DILocation(line: 290, column: 1, scope: !164)
