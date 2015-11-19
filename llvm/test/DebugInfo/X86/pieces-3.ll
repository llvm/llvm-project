; RUN: llc %s -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s
;
;    // Compile with -O1
;    typedef struct {
;      int a;
;      int b;
;    } Inner;
;
;    typedef struct {
;      Inner inner[2];
;    } Outer;
;
;    int foo(Outer outer) {
;      Inner i1 = outer.inner[1];
;      return i1.a;
;    }
;
; CHECK: DW_TAG_formal_parameter [3]
; CHECK-NEXT:   DW_AT_location [DW_FORM_data4]        ([[LOC:.*]])
; CHECK-NEXT:   DW_AT_name {{.*}}"outer"
; CHECK: DW_TAG_variable
;                                                 rsi, piece 0x00000004
; CHECK-NEXT:   DW_AT_location [DW_FORM_block1]       {{.*}} 54 93 04
; CHECK-NEXT:   "i1"
;
; CHECK: .debug_loc
; CHECK: [[LOC]]:
; CHECK: Beginning address offset: 0x0000000000000000
; CHECK:    Ending address offset: 0x0000000000000004
; rdi, piece 0x00000008, piece 0x00000004, rsi, piece 0x00000004
; CHECK: Location description: 55 93 08 93 04 54 93 04 
;
; ModuleID = '/Volumes/Data/llvm/test/DebugInfo/X86/sroasplit-2.ll'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; Function Attrs: nounwind ssp uwtable
define i32 @foo(i64 %outer.coerce0, i64 %outer.coerce1) #0 !dbg !4 {
  call void @llvm.dbg.value(metadata i64 %outer.coerce0, i64 0, metadata !24, metadata !25), !dbg !26
  call void @llvm.dbg.declare(metadata !{null}, metadata !27, metadata !28), !dbg !26
  call void @llvm.dbg.value(metadata i64 %outer.coerce1, i64 0, metadata !29, metadata !30), !dbg !26
  call void @llvm.dbg.declare(metadata !{null}, metadata !31, metadata !32), !dbg !26
  %outer.sroa.1.8.extract.trunc = trunc i64 %outer.coerce1 to i32, !dbg !33
  call void @llvm.dbg.value(metadata i32 %outer.sroa.1.8.extract.trunc, i64 0, metadata !34, metadata !35), !dbg !33
  %outer.sroa.1.12.extract.shift = lshr i64 %outer.coerce1, 32, !dbg !33
  %outer.sroa.1.12.extract.trunc = trunc i64 %outer.sroa.1.12.extract.shift to i32, !dbg !33
  call void @llvm.dbg.value(metadata i64 %outer.sroa.1.12.extract.shift, i64 0, metadata !34, metadata !35), !dbg !33
  call void @llvm.dbg.value(metadata i32 %outer.sroa.1.12.extract.trunc, i64 0, metadata !34, metadata !35), !dbg !33
  call void @llvm.dbg.declare(metadata !{null}, metadata !34, metadata !35), !dbg !33
  ret i32 %outer.sroa.1.8.extract.trunc, !dbg !36
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1) #2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable "no-frame-pointer-elim"="true" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !22}
!llvm.ident = !{!23}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "sroasplit-2.c", directory: "")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "foo", line: 10, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 10, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "sroasplit-2.c", directory: "")
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_typedef, name: "Outer", line: 8, file: !1, baseType: !10)
!10 = !DICompositeType(tag: DW_TAG_structure_type, line: 6, size: 128, align: 32, file: !1, elements: !11)
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "inner", line: 7, size: 128, align: 32, file: !1, scope: !10, baseType: !13)
!13 = !DICompositeType(tag: DW_TAG_array_type, size: 128, align: 32, baseType: !14, elements: !19)
!14 = !DIDerivedType(tag: DW_TAG_typedef, name: "Inner", line: 4, file: !1, baseType: !15)
!15 = !DICompositeType(tag: DW_TAG_structure_type, line: 1, size: 64, align: 32, file: !1, elements: !16)
!16 = !{!17, !18}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "a", line: 2, size: 32, align: 32, file: !1, scope: !15, baseType: !8)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "b", line: 3, size: 32, align: 32, offset: 32, file: !1, scope: !15, baseType: !8)
!19 = !{!20}
!20 = !DISubrange(count: 2)
!21 = !{i32 2, !"Dwarf Version", i32 2}
!22 = !{i32 1, !"Debug Info Version", i32 3}
!23 = !{!"clang version 3.5.0 "}
!24 = !DILocalVariable(name: "outer", line: 10, arg: 1, scope: !4, file: !5, type: !9)
!25 = !DIExpression(DW_OP_bit_piece, 0, 64)
!26 = !DILocation(line: 10, scope: !4)
!27 = !DILocalVariable(name: "outer", line: 10, arg: 1, scope: !4, file: !5, type: !9)
!28 = !DIExpression(DW_OP_bit_piece, 64, 64)
!29 = !DILocalVariable(name: "outer", line: 10, arg: 1, scope: !4, file: !5, type: !9)
!30 = !DIExpression(DW_OP_bit_piece, 96, 32)
!31 = !DILocalVariable(name: "outer", line: 10, arg: 1, scope: !4, file: !5, type: !9)
!32 = !DIExpression(DW_OP_bit_piece, 64, 32)
!33 = !DILocation(line: 11, scope: !4)
!34 = !DILocalVariable(name: "i1", line: 11, scope: !4, file: !5, type: !14)
!35 = !DIExpression(DW_OP_bit_piece, 0, 32)
!36 = !DILocation(line: 12, scope: !4)
