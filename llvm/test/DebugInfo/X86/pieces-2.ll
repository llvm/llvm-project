; RUN: llc %s -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s
;
;    // Compile with -O1
;    typedef struct {
;      int a;
;      long int b;
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
;
; CHECK: DW_TAG_variable [4]
; CHECK-NEXT:   DW_AT_location [DW_FORM_data4]        ([[LOC:.*]])
; CHECK-NEXT:  DW_AT_name {{.*}}"i1"
;
; CHECK: .debug_loc
; CHECK: [[LOC]]: Beginning address offset: 0x0000000000000004
; CHECK-NEXT:        Ending address offset: 0x0000000000000005
;                                           rax, piece 0x00000004
; CHECK-NEXT:         Location description: 50 93 04
;
; ModuleID = '/Volumes/Data/llvm/test/DebugInfo/X86/sroasplit-1.ll'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%struct.Outer = type { [2 x %struct.Inner] }
%struct.Inner = type { i32, i64 }

; Function Attrs: nounwind ssp uwtable
define i32 @foo(%struct.Outer* byval align 8 %outer) #0 !dbg !4 {
entry:
  call void @llvm.dbg.declare(metadata %struct.Outer* %outer, metadata !25, metadata !DIExpression()), !dbg !26
  %i1.sroa.0.0..sroa_idx = getelementptr inbounds %struct.Outer, %struct.Outer* %outer, i64 0, i32 0, i64 1, i32 0, !dbg !27
  %i1.sroa.0.0.copyload = load i32, i32* %i1.sroa.0.0..sroa_idx, align 8, !dbg !27
  call void @llvm.dbg.value(metadata i32 %i1.sroa.0.0.copyload, i64 0, metadata !28, metadata !29), !dbg !27
  %i1.sroa.2.0..sroa_raw_cast = bitcast %struct.Outer* %outer to i8*, !dbg !27
  %i1.sroa.2.0..sroa_raw_idx = getelementptr inbounds i8, i8* %i1.sroa.2.0..sroa_raw_cast, i64 20, !dbg !27
  ret i32 %i1.sroa.0.0.copyload, !dbg !32
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22, !23}
!llvm.ident = !{!24}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "sroasplit-1.c", directory: "")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", line: 10, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 10, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "sroasplit-1.c", directory: "")
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_typedef, name: "Outer", line: 8, file: !1, baseType: !10)
!10 = !DICompositeType(tag: DW_TAG_structure_type, line: 6, size: 256, align: 64, file: !1, elements: !11)
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "inner", line: 7, size: 256, align: 64, file: !1, scope: !10, baseType: !13)
!13 = !DICompositeType(tag: DW_TAG_array_type, size: 256, align: 64, baseType: !14, elements: !20)
!14 = !DIDerivedType(tag: DW_TAG_typedef, name: "Inner", line: 4, file: !1, baseType: !15)
!15 = !DICompositeType(tag: DW_TAG_structure_type, line: 1, size: 128, align: 64, file: !1, elements: !16)
!16 = !{!17, !18}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "a", line: 2, size: 32, align: 32, file: !1, scope: !15, baseType: !8)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "b", line: 3, size: 64, align: 64, offset: 64, file: !1, scope: !15, baseType: !19)
!19 = !DIBasicType(tag: DW_TAG_base_type, name: "long int", size: 64, align: 64, encoding: DW_ATE_signed)
!20 = !{!21}
!21 = !DISubrange(count: 2)
!22 = !{i32 2, !"Dwarf Version", i32 2}
!23 = !{i32 1, !"Debug Info Version", i32 3}
!24 = !{!"clang version 3.5.0 "}
!25 = !DILocalVariable(name: "outer", line: 10, arg: 1, scope: !4, file: !5, type: !9)
!26 = !DILocation(line: 10, scope: !4)
!27 = !DILocation(line: 11, scope: !4)
!28 = !DILocalVariable(name: "i1", line: 11, scope: !4, file: !5, type: !14)
!29 = !DIExpression(DW_OP_bit_piece, 0, 32)
!31 = !{i32 3, i32 0, i32 12}
!32 = !DILocation(line: 12, scope: !4)
