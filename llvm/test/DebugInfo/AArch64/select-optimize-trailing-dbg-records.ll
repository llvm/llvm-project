; RUN: opt %s -passes='require<profile-summary>,function(select-optimize)' -o - -S \
; RUN: | FileCheck %s
; RUN: opt %s --try-experimental-debuginfo-iterators -passes='require<profile-summary>,function(select-optimize)' -o - -S \
; RUN: | FileCheck %s

;; Check that the dbg.value is moved into the start of the end-block of the
;; inserted if-block.

; CHECK: select.end:
; CHECK-NEXT: %[[PHI:.*]] = phi i32
; CHECK-NEXT: #dbg_value(i32 %[[PHI]],

source_filename = "test.ll"
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-fuchsia"

%struct.hb_glyph_info_t = type { i32, i32, i32, %union._hb_var_int_t, %union._hb_var_int_t }
%union._hb_var_int_t = type { i32 }

define void @_Z22_hb_ot_shape_normalizePK18hb_ot_shape_plan_tP11hb_buffer_tP9hb_font_t() {
entry:
  br label %while.body193

while.body193:                                    ; preds = %while.body193, %entry
  %starter.0337 = phi i32 [ %spec.select322, %while.body193 ], [ 0, %entry ]
  %idxprom207 = zext i32 %starter.0337 to i64
  %arrayidx208 = getelementptr %struct.hb_glyph_info_t, ptr null, i64 %idxprom207
  %0 = load i32, ptr %arrayidx208, align 4
  %call247.val = load i16, ptr null, align 4
  %cmp249327 = icmp ult i16 %call247.val, 0
  %cmp249 = select i1 false, i1 false, i1 %cmp249327
  %spec.select322 = select i1 %cmp249, i32 0, i32 %starter.0337
  tail call void @llvm.dbg.value(metadata i32 %spec.select322, metadata !13, metadata !DIExpression()), !dbg !20
  br label %while.body193
}

declare void @llvm.dbg.value(metadata, metadata, metadata)


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !3, imports: !2, splitDebugInlining: false, nameTableKind: GNU)
!1 = !DIFile(filename: "../../third_party/harfbuzz-ng/src/src/hb-ot-shape-normalize.cc", directory: ".")
!2 = !{}
!3 = !{!4, !9}
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!5 = distinct !DIGlobalVariable(scope: null, file: !1, line: 383, type: !6, isLocal: true, isDefinition: true)
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 112, elements: !2)
!7 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !8)
!8 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_unsigned_char)
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = distinct !DIGlobalVariable(scope: null, file: !1, line: 410, type: !11, isLocal: true, isDefinition: true)
!11 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 96, elements: !2)
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !DILocalVariable(name: "starter", scope: !14, file: !1, line: 441, type: !19)
!14 = distinct !DILexicalBlock(scope: !15, file: !1, line: 435, column: 3)
!15 = distinct !DILexicalBlock(scope: !16, file: !1, line: 431, column: 7)
!16 = distinct !DISubprogram(name: "_hb_ot_shape_normalize", linkageName: "_Z22_hb_ot_shape_normalizePK18hb_ot_shape_plan_tP11hb_buffer_tP9hb_font_t", scope: !1, file: !1, line: 291, type: !17, scopeLine: 294, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!17 = distinct !DISubroutineType(types: !18)
!18 = !{null}
!19 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!20 = !DILocation(line: 0, scope: !14)
