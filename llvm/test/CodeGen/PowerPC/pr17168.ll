; RUN: llc -verify-machineinstrs -mcpu=pwr7 -O0 < %s

; This test formerly failed due to a DBG_VALUE being placed prior to a PHI
; when fast-isel is partially successful before punting to DAG-isel.

source_filename = "test/CodeGen/PowerPC/pr17168.ll"
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@grid_points = external global [3 x i32], align 4, !dbg !0

; Function Attrs: nounwind
define fastcc void @compute_rhs() #0 !dbg !263 {
entry:
  br i1 undef, label %for.cond871.preheader.for.inc960_crit_edge, label %for.end1042, !dbg !281

for.cond871.preheader.for.inc960_crit_edge:       ; preds = %for.cond871.preheader.for.inc960_crit_edge, %entry
  br i1 false, label %for.cond871.preheader.for.inc960_crit_edge, label %for.cond964.preheader, !dbg !281

for.cond964.preheader:                            ; preds = %for.cond871.preheader.for.inc960_crit_edge
  br i1 undef, label %for.cond968.preheader, label %for.end1042, !dbg !283

for.cond968.preheader:                            ; preds = %for.cond968.preheader, %for.cond964.preheader
  br i1 false, label %for.cond968.preheader, label %for.end1042, !dbg !283

for.end1042:                                      ; preds = %for.cond968.preheader, %for.cond964.preheader, %entry

  %0 = phi i32 [ undef, %for.cond964.preheader ], [ undef, %for.cond968.preheader ], [ undef, %entry ]
  %1 = load i32, ptr @grid_points, align 4, !dbg !285, !tbaa !286
  tail call void @llvm.dbg.value(metadata i32 1, i64 0, metadata !268, metadata !290), !dbg !291
  %sub10454270 = add nsw i32 %0, -1, !dbg !291
  %cmp10464271 = icmp sgt i32 %sub10454270, 1, !dbg !291
  %sub11134263 = add nsw i32 %1, -1, !dbg !293
  %cmp11144264 = icmp sgt i32 %sub11134263, 1, !dbg !293
  br i1 %cmp11144264, label %for.cond1116.preheader, label %for.cond1816.preheader.for.inc1898_crit_edge, !dbg !293

for.cond1116.preheader:                           ; preds = %for.inc1658, %for.end1042
  br i1 %cmp10464271, label %for.body1123, label %for.inc1658, !dbg !295

for.body1123:                                     ; preds = %for.body1123, %for.cond1116.preheader

  br label %for.body1123, !dbg !298

for.inc1658:                                      ; preds = %for.cond1116.preheader
  br i1 undef, label %for.cond1116.preheader, label %for.cond1816.preheader.for.inc1898_crit_edge, !dbg !293

for.cond1816.preheader.for.inc1898_crit_edge:     ; preds = %for.cond1816.preheader.for.inc1898_crit_edge, %for.inc1658, %for.end1042
  br label %for.cond1816.preheader.for.inc1898_crit_edge, !dbg !301
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!7}
!llvm.module.flags = !{!261, !262}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "grid_points", scope: null, file: !2, line: 28, type: !3, isLocal: true, isDefinition: true)
!2 = !DIFile(filename: "./header.h", directory: "/home/hfinkel/src/NPB2.3-omp-C/BT")
!3 = !DICompositeType(tag: DW_TAG_array_type, baseType: !4, size: 96, align: 32, elements: !5)
!4 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!5 = !{!6}
!6 = !DISubrange(count: 3)
!7 = distinct !DICompileUnit(language: DW_LANG_C99, file: !8, producer: "clang version 3.4 (trunk 190311)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !9, retainedTypes: !9, globals: !10, imports: !9)
!8 = !DIFile(filename: "bt.c", directory: "/home/hfinkel/src/NPB2.3-omp-C/BT")
!9 = !{}
!10 = !{!0, !11, !14, !20, !22, !24, !26, !28, !30, !32, !34, !36, !38, !40, !42, !44, !46, !48, !50, !52, !54, !56, !58, !60, !62, !64, !66, !68, !70, !72, !74, !76, !78, !80, !82, !84, !86, !88, !93, !97, !99, !101, !103, !105, !107, !109, !114, !116, !118, !120, !122, !124, !126, !128, !130, !132, !134, !136, !138, !140, !142, !144, !146, !148, !150, !152, !154, !156, !158, !160, !162, !164, !166, !168, !170, !172, !174, !176, !178, !180, !182, !184, !186, !188, !190, !192, !194, !196, !198, !200, !202, !204, !206, !208, !210, !212, !214, !216, !218, !220, !222, !224, !226, !228, !230, !232, !236, !241, !243, !247, !249, !253, !255, !257, !259}
!11 = !DIGlobalVariableExpression(var: !12, expr: !DIExpression())
!12 = !DIGlobalVariable(name: "dt", scope: null, file: !2, line: 35, type: !13, isLocal: true, isDefinition: true)
!13 = !DIBasicType(name: "double", size: 64, align: 64, encoding: DW_ATE_float)
!14 = !DIGlobalVariableExpression(var: !15, expr: !DIExpression())
!15 = !DIGlobalVariable(name: "rhs", scope: null, file: !2, line: 68, type: !16, isLocal: true, isDefinition: true)
!16 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 1385839040, align: 64, elements: !17)
!17 = !{!18, !18, !18, !19}
!18 = !DISubrange(count: 163)
!19 = !DISubrange(count: 5)
!20 = !DIGlobalVariableExpression(var: !21, expr: !DIExpression())
!21 = !DIGlobalVariable(name: "zzcon5", scope: null, file: !2, line: 42, type: !13, isLocal: true, isDefinition: true)
!22 = !DIGlobalVariableExpression(var: !23, expr: !DIExpression())
!23 = !DIGlobalVariable(name: "zzcon4", scope: null, file: !2, line: 42, type: !13, isLocal: true, isDefinition: true)
!24 = !DIGlobalVariableExpression(var: !25, expr: !DIExpression())
!25 = !DIGlobalVariable(name: "zzcon3", scope: null, file: !2, line: 42, type: !13, isLocal: true, isDefinition: true)
!26 = !DIGlobalVariableExpression(var: !27, expr: !DIExpression())
!27 = !DIGlobalVariable(name: "dz5tz1", scope: null, file: !2, line: 43, type: !13, isLocal: true, isDefinition: true)
!28 = !DIGlobalVariableExpression(var: !29, expr: !DIExpression())
!29 = !DIGlobalVariable(name: "dz4tz1", scope: null, file: !2, line: 43, type: !13, isLocal: true, isDefinition: true)
!30 = !DIGlobalVariableExpression(var: !31, expr: !DIExpression())
!31 = !DIGlobalVariable(name: "dz3tz1", scope: null, file: !2, line: 43, type: !13, isLocal: true, isDefinition: true)
!32 = !DIGlobalVariableExpression(var: !33, expr: !DIExpression())
!33 = !DIGlobalVariable(name: "zzcon2", scope: null, file: !2, line: 42, type: !13, isLocal: true, isDefinition: true)
!34 = !DIGlobalVariableExpression(var: !35, expr: !DIExpression())
!35 = !DIGlobalVariable(name: "dz2tz1", scope: null, file: !2, line: 43, type: !13, isLocal: true, isDefinition: true)
!36 = !DIGlobalVariableExpression(var: !37, expr: !DIExpression())
!37 = !DIGlobalVariable(name: "tz2", scope: null, file: !2, line: 31, type: !13, isLocal: true, isDefinition: true)
!38 = !DIGlobalVariableExpression(var: !39, expr: !DIExpression())
!39 = !DIGlobalVariable(name: "dz1tz1", scope: null, file: !2, line: 43, type: !13, isLocal: true, isDefinition: true)
!40 = !DIGlobalVariableExpression(var: !41, expr: !DIExpression())
!41 = !DIGlobalVariable(name: "yycon5", scope: null, file: !2, line: 40, type: !13, isLocal: true, isDefinition: true)
!42 = !DIGlobalVariableExpression(var: !43, expr: !DIExpression())
!43 = !DIGlobalVariable(name: "yycon4", scope: null, file: !2, line: 40, type: !13, isLocal: true, isDefinition: true)
!44 = !DIGlobalVariableExpression(var: !45, expr: !DIExpression())
!45 = !DIGlobalVariable(name: "yycon3", scope: null, file: !2, line: 40, type: !13, isLocal: true, isDefinition: true)
!46 = !DIGlobalVariableExpression(var: !47, expr: !DIExpression())
!47 = !DIGlobalVariable(name: "dy5ty1", scope: null, file: !2, line: 41, type: !13, isLocal: true, isDefinition: true)
!48 = !DIGlobalVariableExpression(var: !49, expr: !DIExpression())
!49 = !DIGlobalVariable(name: "dy4ty1", scope: null, file: !2, line: 41, type: !13, isLocal: true, isDefinition: true)
!50 = !DIGlobalVariableExpression(var: !51, expr: !DIExpression())
!51 = !DIGlobalVariable(name: "dy3ty1", scope: null, file: !2, line: 41, type: !13, isLocal: true, isDefinition: true)
!52 = !DIGlobalVariableExpression(var: !53, expr: !DIExpression())
!53 = !DIGlobalVariable(name: "yycon2", scope: null, file: !2, line: 40, type: !13, isLocal: true, isDefinition: true)
!54 = !DIGlobalVariableExpression(var: !55, expr: !DIExpression())
!55 = !DIGlobalVariable(name: "dy2ty1", scope: null, file: !2, line: 41, type: !13, isLocal: true, isDefinition: true)
!56 = !DIGlobalVariableExpression(var: !57, expr: !DIExpression())
!57 = !DIGlobalVariable(name: "ty2", scope: null, file: !2, line: 31, type: !13, isLocal: true, isDefinition: true)
!58 = !DIGlobalVariableExpression(var: !59, expr: !DIExpression())
!59 = !DIGlobalVariable(name: "dy1ty1", scope: null, file: !2, line: 41, type: !13, isLocal: true, isDefinition: true)
!60 = !DIGlobalVariableExpression(var: !61, expr: !DIExpression())
!61 = !DIGlobalVariable(name: "dssp", scope: null, file: !2, line: 35, type: !13, isLocal: true, isDefinition: true)
!62 = !DIGlobalVariableExpression(var: !63, expr: !DIExpression())
!63 = !DIGlobalVariable(name: "c1", scope: null, file: !2, line: 45, type: !13, isLocal: true, isDefinition: true)
!64 = !DIGlobalVariableExpression(var: !65, expr: !DIExpression())
!65 = !DIGlobalVariable(name: "xxcon5", scope: null, file: !2, line: 38, type: !13, isLocal: true, isDefinition: true)
!66 = !DIGlobalVariableExpression(var: !67, expr: !DIExpression())
!67 = !DIGlobalVariable(name: "xxcon4", scope: null, file: !2, line: 38, type: !13, isLocal: true, isDefinition: true)
!68 = !DIGlobalVariableExpression(var: !69, expr: !DIExpression())
!69 = !DIGlobalVariable(name: "xxcon3", scope: null, file: !2, line: 38, type: !13, isLocal: true, isDefinition: true)
!70 = !DIGlobalVariableExpression(var: !71, expr: !DIExpression())
!71 = !DIGlobalVariable(name: "dx5tx1", scope: null, file: !2, line: 39, type: !13, isLocal: true, isDefinition: true)
!72 = !DIGlobalVariableExpression(var: !73, expr: !DIExpression())
!73 = !DIGlobalVariable(name: "dx4tx1", scope: null, file: !2, line: 39, type: !13, isLocal: true, isDefinition: true)
!74 = !DIGlobalVariableExpression(var: !75, expr: !DIExpression())
!75 = !DIGlobalVariable(name: "dx3tx1", scope: null, file: !2, line: 39, type: !13, isLocal: true, isDefinition: true)
!76 = !DIGlobalVariableExpression(var: !77, expr: !DIExpression())
!77 = !DIGlobalVariable(name: "c2", scope: null, file: !2, line: 45, type: !13, isLocal: true, isDefinition: true)
!78 = !DIGlobalVariableExpression(var: !79, expr: !DIExpression())
!79 = !DIGlobalVariable(name: "con43", scope: null, file: !2, line: 48, type: !13, isLocal: true, isDefinition: true)
!80 = !DIGlobalVariableExpression(var: !81, expr: !DIExpression())
!81 = !DIGlobalVariable(name: "xxcon2", scope: null, file: !2, line: 38, type: !13, isLocal: true, isDefinition: true)
!82 = !DIGlobalVariableExpression(var: !83, expr: !DIExpression())
!83 = !DIGlobalVariable(name: "dx2tx1", scope: null, file: !2, line: 39, type: !13, isLocal: true, isDefinition: true)
!84 = !DIGlobalVariableExpression(var: !85, expr: !DIExpression())
!85 = !DIGlobalVariable(name: "tx2", scope: null, file: !2, line: 31, type: !13, isLocal: true, isDefinition: true)
!86 = !DIGlobalVariableExpression(var: !87, expr: !DIExpression())
!87 = !DIGlobalVariable(name: "dx1tx1", scope: null, file: !2, line: 39, type: !13, isLocal: true, isDefinition: true)
!88 = !DIGlobalVariableExpression(var: !89, expr: !DIExpression())
!89 = !DIGlobalVariable(name: "forcing", scope: null, file: !2, line: 66, type: !90, isLocal: true, isDefinition: true)
!90 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 1663006848, align: 64, elements: !91)
!91 = !{!18, !18, !18, !92}
!92 = !DISubrange(count: 6)
!93 = !DIGlobalVariableExpression(var: !94, expr: !DIExpression())
!94 = !DIGlobalVariable(name: "qs", scope: null, file: !2, line: 63, type: !95, isLocal: true, isDefinition: true)
!95 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 277167808, align: 64, elements: !96)
!96 = !{!18, !18, !18}
!97 = !DIGlobalVariableExpression(var: !98, expr: !DIExpression())
!98 = !DIGlobalVariable(name: "square", scope: null, file: !2, line: 65, type: !95, isLocal: true, isDefinition: true)
!99 = !DIGlobalVariableExpression(var: !100, expr: !DIExpression())
!100 = !DIGlobalVariable(name: "ws", scope: null, file: !2, line: 62, type: !95, isLocal: true, isDefinition: true)
!101 = !DIGlobalVariableExpression(var: !102, expr: !DIExpression())
!102 = !DIGlobalVariable(name: "vs", scope: null, file: !2, line: 61, type: !95, isLocal: true, isDefinition: true)
!103 = !DIGlobalVariableExpression(var: !104, expr: !DIExpression())
!104 = !DIGlobalVariable(name: "us", scope: null, file: !2, line: 60, type: !95, isLocal: true, isDefinition: true)
!105 = !DIGlobalVariableExpression(var: !106, expr: !DIExpression())
!106 = !DIGlobalVariable(name: "rho_i", scope: null, file: !2, line: 64, type: !95, isLocal: true, isDefinition: true)
!107 = !DIGlobalVariableExpression(var: !108, expr: !DIExpression())
!108 = !DIGlobalVariable(name: "u", scope: null, file: !2, line: 67, type: !16, isLocal: true, isDefinition: true)
!109 = !DIGlobalVariableExpression(var: !110, expr: !DIExpression())
!110 = !DIGlobalVariable(name: "ce", scope: null, file: !2, line: 36, type: !111, isLocal: true, isDefinition: true)
!111 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 4160, align: 64, elements: !112)
!112 = !{!19, !113}
!113 = !DISubrange(count: 13)
!114 = !DIGlobalVariableExpression(var: !115, expr: !DIExpression())
!115 = !DIGlobalVariable(name: "dnzm1", scope: null, file: !2, line: 44, type: !13, isLocal: true, isDefinition: true)
!116 = !DIGlobalVariableExpression(var: !117, expr: !DIExpression())
!117 = !DIGlobalVariable(name: "dnym1", scope: null, file: !2, line: 44, type: !13, isLocal: true, isDefinition: true)
!118 = !DIGlobalVariableExpression(var: !119, expr: !DIExpression())
!119 = !DIGlobalVariable(name: "dnxm1", scope: null, file: !2, line: 44, type: !13, isLocal: true, isDefinition: true)
!120 = !DIGlobalVariableExpression(var: !121, expr: !DIExpression())
!121 = !DIGlobalVariable(name: "zzcon1", scope: null, file: !2, line: 42, type: !13, isLocal: true, isDefinition: true)
!122 = !DIGlobalVariableExpression(var: !123, expr: !DIExpression())
!123 = !DIGlobalVariable(name: "yycon1", scope: null, file: !2, line: 40, type: !13, isLocal: true, isDefinition: true)
!124 = !DIGlobalVariableExpression(var: !125, expr: !DIExpression())
!125 = !DIGlobalVariable(name: "xxcon1", scope: null, file: !2, line: 38, type: !13, isLocal: true, isDefinition: true)
!126 = !DIGlobalVariableExpression(var: !127, expr: !DIExpression())
!127 = !DIGlobalVariable(name: "con16", scope: null, file: !2, line: 48, type: !13, isLocal: true, isDefinition: true)
!128 = !DIGlobalVariableExpression(var: !129, expr: !DIExpression())
!129 = !DIGlobalVariable(name: "c2iv", scope: null, file: !2, line: 48, type: !13, isLocal: true, isDefinition: true)
!130 = !DIGlobalVariableExpression(var: !131, expr: !DIExpression())
!131 = !DIGlobalVariable(name: "c3c4tz3", scope: null, file: !2, line: 48, type: !13, isLocal: true, isDefinition: true)
!132 = !DIGlobalVariableExpression(var: !133, expr: !DIExpression())
!133 = !DIGlobalVariable(name: "c3c4ty3", scope: null, file: !2, line: 48, type: !13, isLocal: true, isDefinition: true)
!134 = !DIGlobalVariableExpression(var: !135, expr: !DIExpression())
!135 = !DIGlobalVariable(name: "c3c4tx3", scope: null, file: !2, line: 48, type: !13, isLocal: true, isDefinition: true)
!136 = !DIGlobalVariableExpression(var: !137, expr: !DIExpression())
!137 = !DIGlobalVariable(name: "comz6", scope: null, file: !2, line: 47, type: !13, isLocal: true, isDefinition: true)
!138 = !DIGlobalVariableExpression(var: !139, expr: !DIExpression())
!139 = !DIGlobalVariable(name: "comz5", scope: null, file: !2, line: 47, type: !13, isLocal: true, isDefinition: true)
!140 = !DIGlobalVariableExpression(var: !141, expr: !DIExpression())
!141 = !DIGlobalVariable(name: "comz4", scope: null, file: !2, line: 47, type: !13, isLocal: true, isDefinition: true)
!142 = !DIGlobalVariableExpression(var: !143, expr: !DIExpression())
!143 = !DIGlobalVariable(name: "comz1", scope: null, file: !2, line: 47, type: !13, isLocal: true, isDefinition: true)
!144 = !DIGlobalVariableExpression(var: !145, expr: !DIExpression())
!145 = !DIGlobalVariable(name: "dtdssp", scope: null, file: !2, line: 45, type: !13, isLocal: true, isDefinition: true)
!146 = !DIGlobalVariableExpression(var: !147, expr: !DIExpression())
!147 = !DIGlobalVariable(name: "c2dttz1", scope: null, file: !2, line: 47, type: !13, isLocal: true, isDefinition: true)
!148 = !DIGlobalVariableExpression(var: !149, expr: !DIExpression())
!149 = !DIGlobalVariable(name: "c2dtty1", scope: null, file: !2, line: 47, type: !13, isLocal: true, isDefinition: true)
!150 = !DIGlobalVariableExpression(var: !151, expr: !DIExpression())
!151 = !DIGlobalVariable(name: "c2dttx1", scope: null, file: !2, line: 47, type: !13, isLocal: true, isDefinition: true)
!152 = !DIGlobalVariableExpression(var: !153, expr: !DIExpression())
!153 = !DIGlobalVariable(name: "dttz2", scope: null, file: !2, line: 46, type: !13, isLocal: true, isDefinition: true)
!154 = !DIGlobalVariableExpression(var: !155, expr: !DIExpression())
!155 = !DIGlobalVariable(name: "dttz1", scope: null, file: !2, line: 46, type: !13, isLocal: true, isDefinition: true)
!156 = !DIGlobalVariableExpression(var: !157, expr: !DIExpression())
!157 = !DIGlobalVariable(name: "dtty2", scope: null, file: !2, line: 46, type: !13, isLocal: true, isDefinition: true)
!158 = !DIGlobalVariableExpression(var: !159, expr: !DIExpression())
!159 = !DIGlobalVariable(name: "dtty1", scope: null, file: !2, line: 46, type: !13, isLocal: true, isDefinition: true)
!160 = !DIGlobalVariableExpression(var: !161, expr: !DIExpression())
!161 = !DIGlobalVariable(name: "dttx2", scope: null, file: !2, line: 46, type: !13, isLocal: true, isDefinition: true)
!162 = !DIGlobalVariableExpression(var: !163, expr: !DIExpression())
!163 = !DIGlobalVariable(name: "dttx1", scope: null, file: !2, line: 46, type: !13, isLocal: true, isDefinition: true)
!164 = !DIGlobalVariableExpression(var: !165, expr: !DIExpression())
!165 = !DIGlobalVariable(name: "c5dssp", scope: null, file: !2, line: 45, type: !13, isLocal: true, isDefinition: true)
!166 = !DIGlobalVariableExpression(var: !167, expr: !DIExpression())
!167 = !DIGlobalVariable(name: "c4dssp", scope: null, file: !2, line: 45, type: !13, isLocal: true, isDefinition: true)
!168 = !DIGlobalVariableExpression(var: !169, expr: !DIExpression())
!169 = !DIGlobalVariable(name: "dzmax", scope: null, file: !2, line: 37, type: !13, isLocal: true, isDefinition: true)
!170 = !DIGlobalVariableExpression(var: !171, expr: !DIExpression())
!171 = !DIGlobalVariable(name: "dymax", scope: null, file: !2, line: 37, type: !13, isLocal: true, isDefinition: true)
!172 = !DIGlobalVariableExpression(var: !173, expr: !DIExpression())
!173 = !DIGlobalVariable(name: "dxmax", scope: null, file: !2, line: 37, type: !13, isLocal: true, isDefinition: true)
!174 = !DIGlobalVariableExpression(var: !175, expr: !DIExpression())
!175 = !DIGlobalVariable(name: "dz5", scope: null, file: !2, line: 34, type: !13, isLocal: true, isDefinition: true)
!176 = !DIGlobalVariableExpression(var: !177, expr: !DIExpression())
!177 = !DIGlobalVariable(name: "dz4", scope: null, file: !2, line: 34, type: !13, isLocal: true, isDefinition: true)
!178 = !DIGlobalVariableExpression(var: !179, expr: !DIExpression())
!179 = !DIGlobalVariable(name: "dz3", scope: null, file: !2, line: 34, type: !13, isLocal: true, isDefinition: true)
!180 = !DIGlobalVariableExpression(var: !181, expr: !DIExpression())
!181 = !DIGlobalVariable(name: "dz2", scope: null, file: !2, line: 34, type: !13, isLocal: true, isDefinition: true)
!182 = !DIGlobalVariableExpression(var: !183, expr: !DIExpression())
!183 = !DIGlobalVariable(name: "dz1", scope: null, file: !2, line: 34, type: !13, isLocal: true, isDefinition: true)
!184 = !DIGlobalVariableExpression(var: !185, expr: !DIExpression())
!185 = !DIGlobalVariable(name: "dy5", scope: null, file: !2, line: 33, type: !13, isLocal: true, isDefinition: true)
!186 = !DIGlobalVariableExpression(var: !187, expr: !DIExpression())
!187 = !DIGlobalVariable(name: "dy4", scope: null, file: !2, line: 33, type: !13, isLocal: true, isDefinition: true)
!188 = !DIGlobalVariableExpression(var: !189, expr: !DIExpression())
!189 = !DIGlobalVariable(name: "dy3", scope: null, file: !2, line: 33, type: !13, isLocal: true, isDefinition: true)
!190 = !DIGlobalVariableExpression(var: !191, expr: !DIExpression())
!191 = !DIGlobalVariable(name: "dy2", scope: null, file: !2, line: 33, type: !13, isLocal: true, isDefinition: true)
!192 = !DIGlobalVariableExpression(var: !193, expr: !DIExpression())
!193 = !DIGlobalVariable(name: "dy1", scope: null, file: !2, line: 33, type: !13, isLocal: true, isDefinition: true)
!194 = !DIGlobalVariableExpression(var: !195, expr: !DIExpression())
!195 = !DIGlobalVariable(name: "dx5", scope: null, file: !2, line: 32, type: !13, isLocal: true, isDefinition: true)
!196 = !DIGlobalVariableExpression(var: !197, expr: !DIExpression())
!197 = !DIGlobalVariable(name: "dx4", scope: null, file: !2, line: 32, type: !13, isLocal: true, isDefinition: true)
!198 = !DIGlobalVariableExpression(var: !199, expr: !DIExpression())
!199 = !DIGlobalVariable(name: "dx3", scope: null, file: !2, line: 32, type: !13, isLocal: true, isDefinition: true)
!200 = !DIGlobalVariableExpression(var: !201, expr: !DIExpression())
!201 = !DIGlobalVariable(name: "dx2", scope: null, file: !2, line: 32, type: !13, isLocal: true, isDefinition: true)
!202 = !DIGlobalVariableExpression(var: !203, expr: !DIExpression())
!203 = !DIGlobalVariable(name: "dx1", scope: null, file: !2, line: 32, type: !13, isLocal: true, isDefinition: true)
!204 = !DIGlobalVariableExpression(var: !205, expr: !DIExpression())
!205 = !DIGlobalVariable(name: "tz3", scope: null, file: !2, line: 31, type: !13, isLocal: true, isDefinition: true)
!206 = !DIGlobalVariableExpression(var: !207, expr: !DIExpression())
!207 = !DIGlobalVariable(name: "tz1", scope: null, file: !2, line: 31, type: !13, isLocal: true, isDefinition: true)
!208 = !DIGlobalVariableExpression(var: !209, expr: !DIExpression())
!209 = !DIGlobalVariable(name: "ty3", scope: null, file: !2, line: 31, type: !13, isLocal: true, isDefinition: true)
!210 = !DIGlobalVariableExpression(var: !211, expr: !DIExpression())
!211 = !DIGlobalVariable(name: "ty1", scope: null, file: !2, line: 31, type: !13, isLocal: true, isDefinition: true)
!212 = !DIGlobalVariableExpression(var: !213, expr: !DIExpression())
!213 = !DIGlobalVariable(name: "tx3", scope: null, file: !2, line: 31, type: !13, isLocal: true, isDefinition: true)
!214 = !DIGlobalVariableExpression(var: !215, expr: !DIExpression())
!215 = !DIGlobalVariable(name: "tx1", scope: null, file: !2, line: 31, type: !13, isLocal: true, isDefinition: true)
!216 = !DIGlobalVariableExpression(var: !217, expr: !DIExpression())
!217 = !DIGlobalVariable(name: "conz1", scope: null, file: !2, line: 45, type: !13, isLocal: true, isDefinition: true)
!218 = !DIGlobalVariableExpression(var: !219, expr: !DIExpression())
!219 = !DIGlobalVariable(name: "c1345", scope: null, file: !2, line: 44, type: !13, isLocal: true, isDefinition: true)
!220 = !DIGlobalVariableExpression(var: !221, expr: !DIExpression())
!221 = !DIGlobalVariable(name: "c3c4", scope: null, file: !2, line: 44, type: !13, isLocal: true, isDefinition: true)
!222 = !DIGlobalVariableExpression(var: !223, expr: !DIExpression())
!223 = !DIGlobalVariable(name: "c1c5", scope: null, file: !2, line: 44, type: !13, isLocal: true, isDefinition: true)
!224 = !DIGlobalVariableExpression(var: !225, expr: !DIExpression())
!225 = !DIGlobalVariable(name: "c1c2", scope: null, file: !2, line: 44, type: !13, isLocal: true, isDefinition: true)
!226 = !DIGlobalVariableExpression(var: !227, expr: !DIExpression())
!227 = !DIGlobalVariable(name: "c5", scope: null, file: !2, line: 45, type: !13, isLocal: true, isDefinition: true)
!228 = !DIGlobalVariableExpression(var: !229, expr: !DIExpression())
!229 = !DIGlobalVariable(name: "c4", scope: null, file: !2, line: 45, type: !13, isLocal: true, isDefinition: true)
!230 = !DIGlobalVariableExpression(var: !231, expr: !DIExpression())
!231 = !DIGlobalVariable(name: "c3", scope: null, file: !2, line: 45, type: !13, isLocal: true, isDefinition: true)
!232 = !DIGlobalVariableExpression(var: !233, expr: !DIExpression())
!233 = !DIGlobalVariable(name: "lhs", scope: null, file: !2, line: 69, type: !234, isLocal: true, isDefinition: true)
!234 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 20787585600, align: 64, elements: !235)
!235 = !{!18, !18, !18, !6, !19, !19}
!236 = !DIGlobalVariableExpression(var: !237, expr: !DIExpression())
!237 = !DIGlobalVariable(name: "q", scope: null, file: !2, line: 73, type: !238, isLocal: true, isDefinition: true)
!238 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 10368, align: 64, elements: !239)
!239 = !{!240}
!240 = !DISubrange(count: 162)
!241 = !DIGlobalVariableExpression(var: !242, expr: !DIExpression())
!242 = !DIGlobalVariable(name: "cuf", scope: null, file: !2, line: 72, type: !238, isLocal: true, isDefinition: true)
!243 = !DIGlobalVariableExpression(var: !244, expr: !DIExpression())
!244 = !DIGlobalVariable(name: "buf", scope: null, file: !2, line: 75, type: !245, isLocal: true, isDefinition: true)
!245 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 51840, align: 64, elements: !246)
!246 = !{!240, !19}
!247 = !DIGlobalVariableExpression(var: !248, expr: !DIExpression())
!248 = !DIGlobalVariable(name: "ue", scope: null, file: !2, line: 74, type: !245, isLocal: true, isDefinition: true)
!249 = !DIGlobalVariableExpression(var: !250, expr: !DIExpression())
!250 = !DIGlobalVariable(name: "njac", scope: null, file: !2, line: 86, type: !251, isLocal: true, isDefinition: true)
!251 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 6886684800, align: 64, elements: !252)
!252 = !{!18, !18, !240, !19, !19}
!253 = !DIGlobalVariableExpression(var: !254, expr: !DIExpression())
!254 = !DIGlobalVariable(name: "fjac", scope: null, file: !2, line: 84, type: !251, isLocal: true, isDefinition: true)
!255 = !DIGlobalVariableExpression(var: !256, expr: !DIExpression())
!256 = !DIGlobalVariable(name: "tmp3", scope: null, file: !2, line: 88, type: !13, isLocal: true, isDefinition: true)
!257 = !DIGlobalVariableExpression(var: !258, expr: !DIExpression())
!258 = !DIGlobalVariable(name: "tmp2", scope: null, file: !2, line: 88, type: !13, isLocal: true, isDefinition: true)
!259 = !DIGlobalVariableExpression(var: !260, expr: !DIExpression())
!260 = !DIGlobalVariable(name: "tmp1", scope: null, file: !2, line: 88, type: !13, isLocal: true, isDefinition: true)
!261 = !{i32 2, !"Dwarf Version", i32 4}
!262 = !{i32 1, !"Debug Info Version", i32 3}
!263 = distinct !DISubprogram(name: "compute_rhs", scope: !8, file: !8, line: 1767, type: !264, isLocal: true, isDefinition: true, scopeLine: 1767, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !7, retainedNodes: !266)
!264 = !DISubroutineType(types: !265)
!265 = !{null}
!266 = !{!267, !268, !269, !270, !271, !272, !273, !274, !275, !276, !277, !278, !279, !280}
!267 = !DILocalVariable(name: "i", scope: !263, file: !8, line: 1769, type: !4)
!268 = !DILocalVariable(name: "j", scope: !263, file: !8, line: 1769, type: !4)
!269 = !DILocalVariable(name: "k", scope: !263, file: !8, line: 1769, type: !4)
!270 = !DILocalVariable(name: "m", scope: !263, file: !8, line: 1769, type: !4)
!271 = !DILocalVariable(name: "rho_inv", scope: !263, file: !8, line: 1770, type: !13)
!272 = !DILocalVariable(name: "uijk", scope: !263, file: !8, line: 1770, type: !13)
!273 = !DILocalVariable(name: "up1", scope: !263, file: !8, line: 1770, type: !13)
!274 = !DILocalVariable(name: "um1", scope: !263, file: !8, line: 1770, type: !13)
!275 = !DILocalVariable(name: "vijk", scope: !263, file: !8, line: 1770, type: !13)
!276 = !DILocalVariable(name: "vp1", scope: !263, file: !8, line: 1770, type: !13)
!277 = !DILocalVariable(name: "vm1", scope: !263, file: !8, line: 1770, type: !13)
!278 = !DILocalVariable(name: "wijk", scope: !263, file: !8, line: 1770, type: !13)
!279 = !DILocalVariable(name: "wp1", scope: !263, file: !8, line: 1770, type: !13)
!280 = !DILocalVariable(name: "wm1", scope: !263, file: !8, line: 1770, type: !13)
!281 = !DILocation(line: 1898, scope: !282)
!282 = distinct !DILexicalBlock(scope: !263, file: !8, line: 1898)
!283 = !DILocation(line: 1913, scope: !284)
!284 = distinct !DILexicalBlock(scope: !263, file: !8, line: 1913)
!285 = !DILocation(line: 1923, scope: !263)
!286 = !{!287, !287, i64 0}
!287 = !{!"int", !288}
!288 = !{!"omnipotent char", !289}
!289 = !{!"Simple C/C++ TBAA"}
!290 = !DIExpression()
!291 = !DILocation(line: 1925, scope: !292)
!292 = distinct !DILexicalBlock(scope: !263, file: !8, line: 1925)
!293 = !DILocation(line: 1939, scope: !294)
!294 = distinct !DILexicalBlock(scope: !263, file: !8, line: 1939)
!295 = !DILocation(line: 1940, scope: !296)
!296 = distinct !DILexicalBlock(scope: !297, file: !8, line: 1940)
!297 = distinct !DILexicalBlock(scope: !294, file: !8, line: 1939)
!298 = !DILocation(line: 1941, scope: !299)
!299 = distinct !DILexicalBlock(scope: !300, file: !8, line: 1941)
!300 = distinct !DILexicalBlock(scope: !296, file: !8, line: 1940)
!301 = !DILocation(line: 2020, scope: !302)
!302 = distinct !DILexicalBlock(scope: !303, file: !8, line: 2020)
!303 = distinct !DILexicalBlock(scope: !304, file: !8, line: 2019)
!304 = distinct !DILexicalBlock(scope: !305, file: !8, line: 2019)
!305 = distinct !DILexicalBlock(scope: !306, file: !8, line: 2018)
!306 = distinct !DILexicalBlock(scope: !263, file: !8, line: 2018)

