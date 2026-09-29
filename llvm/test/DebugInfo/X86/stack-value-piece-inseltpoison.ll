; RUN: llc %s -filetype=obj -o - | llvm-dwarfdump -debug-info -debug-loc - | FileCheck %s
; Test that DW_OP_piece is emitted for constants.
;
; // Generated from:
; typedef struct { int a, b; } I;
; I i(int i) {
;   I r = {i, 0};
;   return r;
; }
;
; typedef struct { float a, b; } F;
; F f(float f) {
;   F r = {f, 0};
;   return r;
; }

; CHECK: .debug_info contents:
; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_name ("i")
; CHECK:   DW_TAG_variable
; CHECK-NEXT:   DW_AT_location ([[I:0x[0-9a-f]+]]
; CHECK-NEXT:     [{{.*}}, {{.*}}): DW_OP_reg5 RDI, DW_OP_piece 0x4, DW_OP_lit0, DW_OP_stack_value, DW_OP_piece 0x4)
; CHECK-NEXT:   DW_AT_name ("r")
;
; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_name ("f")
; CHECK:   DW_TAG_variable
; CHECK-NEXT:   DW_AT_location ([[F:0x[0-9a-f]+]]
; CHECK-NEXT:     [{{.*}}, {{.*}}): DW_OP_reg17 XMM0, DW_OP_piece 0x4, {{(DW_OP_lit0, DW_OP_stack_value, DW_OP_piece 0x4|DW_OP_implicit_value 0x4 0x00 0x00 0x00 0x00)}}
; CHECK-NEXT:     [{{.*}}, {{.*}}): DW_OP_piece 0x4, {{(DW_OP_lit0, DW_OP_stack_value, DW_OP_piece 0x4|DW_OP_implicit_value 0x4 0x00 0x00 0x00 0x00)}})
; CHECK-NEXT:   DW_AT_name ("r")
;
; CHECK: .debug_loc contents:
; CHECK:      [[I]]:
; CHECK-NEXT:   ({{.*}}, {{.*}}): DW_OP_reg5 RDI, DW_OP_piece 0x4, DW_OP_lit0, DW_OP_stack_value, DW_OP_piece 0x4
; CHECK:      [[F]]:
; CHECK-NEXT:   ({{.*}}, {{.*}}): DW_OP_reg17 XMM0, DW_OP_piece 0x4, {{DW_OP_lit0, DW_OP_stack_value, DW_OP_piece 0x4|DW_OP_implicit_value 0x4 0x00 0x00 0x00 0x00}}

source_filename = "stack-value-piece.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

%struct.I = type { i32, i32 }
%struct.F = type { float, float }

; Function Attrs: nounwind readnone ssp uwtable
define i64 @i(i32 %i) local_unnamed_addr #0 !dbg !7 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %i, metadata !18, metadata !22), !dbg !21
  tail call void @llvm.dbg.value(metadata i32 0, metadata !18, metadata !23), !dbg !21
  %retval.sroa.0.0.insert.ext = zext i32 %i to i64, !dbg !24
  ret i64 %retval.sroa.0.0.insert.ext, !dbg !24
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone ssp uwtable
define <2 x float> @f(float %f) local_unnamed_addr #0 !dbg !25 {
entry:
  tail call void @llvm.dbg.value(metadata float %f, metadata !36, metadata !22), !dbg !38
  tail call void @llvm.dbg.value(metadata float 0.000000e+00, metadata !36, metadata !23), !dbg !38
  %retval.sroa.0.0.vec.insert = insertelement <2 x float> poison, float %f, i32 0, !dbg !39
  %retval.sroa.0.4.vec.insert = insertelement <2 x float> %retval.sroa.0.0.vec.insert, float 0.000000e+00, i32 1, !dbg !39
  ret <2 x float> %retval.sroa.0.4.vec.insert, !dbg !40
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind readnone ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0 (trunk 285655) (llvm/trunk 285654)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "stack-value-piece.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 4.0.0 (trunk 285655) (llvm/trunk 285654)"}
!7 = distinct !DISubprogram(name: "i", scope: !1, file: !1, line: 2, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !14}
!10 = !DIDerivedType(tag: DW_TAG_typedef, name: "I", file: !1, line: 1, baseType: !11)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !1, line: 1, size: 64, elements: !12)
!12 = !{!13, !15}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !11, file: !1, line: 1, baseType: !14, size: 32)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !11, file: !1, line: 1, baseType: !14, size: 32, offset: 32)
!18 = !DILocalVariable(name: "r", scope: !7, file: !1, line: 3, type: !10)
!19 = !DIExpression()
!20 = !DILocation(line: 2, column: 9, scope: !7)
!21 = !DILocation(line: 3, column: 5, scope: !7)
!22 = !DIExpression(DW_OP_LLVM_fragment, 0, 32)
!23 = !DIExpression(DW_OP_LLVM_fragment, 32, 32)
!24 = !DILocation(line: 5, column: 1, scope: !7)
!25 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 8, type: !26, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: true, unit: !0)
!26 = !DISubroutineType(types: !27)
!27 = !{!28, !32}
!28 = !DIDerivedType(tag: DW_TAG_typedef, name: "F", file: !1, line: 7, baseType: !29)
!29 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !1, line: 7, size: 64, elements: !30)
!30 = !{!31, !33}
!31 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !29, file: !1, line: 7, baseType: !32, size: 32)
!32 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!33 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !29, file: !1, line: 7, baseType: !32, size: 32, offset: 32)
!36 = !DILocalVariable(name: "r", scope: !25, file: !1, line: 9, type: !28)
!37 = !DILocation(line: 8, column: 11, scope: !25)
!38 = !DILocation(line: 9, column: 5, scope: !25)
!39 = !DILocation(line: 10, column: 10, scope: !25)
!40 = !DILocation(line: 11, column: 1, scope: !25)
