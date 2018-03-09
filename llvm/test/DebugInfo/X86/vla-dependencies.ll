; RUN: llc -mtriple=x86_64-apple-darwin %s -o - -filetype=obj | llvm-dwarfdump - | FileCheck %s
; CHECK:  DW_TAG_subprogram
; CHECK:    DW_AT_name	("h")
; CHECK: 0x00000[[VLAEXPR:.*]]:     DW_TAG_variable
; CHECK-NEXT:    DW_AT_name	("vla_expr")
; CHECK:        DW_TAG_array_type
; CHECK-NEXT:     DW_AT_type	{{.*}}"int"
; CHECK-NOT: DW_TAG
; CHECK:        DW_TAG_subrange_type
; CHECK-NEXT:     DW_AT_type {{.*}}"sizetype"
; CHECK-NEXT:     DW_AT_count	(0x00000[[VLAEXPR]]
;
;
; Generated from (and then modified):
;
; #define DECLARE_ARRAY(type, var_name, size) type var_name[size]
;
; void h(void);
; void k(void *);
;
; void g() {
;   h();
; }
;
; void h() {
;   int count = 2;
;   DECLARE_ARRAY(int, array, count);
;   k((void *)array);
; }
source_filename = "/tmp/test.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

; Function Attrs: nounwind ssp uwtable
define void @g() local_unnamed_addr #0 !dbg !10 {
entry:
  %vla2.i = alloca [2 x i32], align 16, !dbg !13
  call void @llvm.dbg.declare(metadata [2 x i32]* %vla2.i, metadata !20, metadata !DIExpression(DW_OP_stack_value)), !dbg !13
  %0 = bitcast [2 x i32]* %vla2.i to i8*, !dbg !25
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0), !dbg !25
  call void @llvm.dbg.value(metadata i32 2, metadata !16, metadata !DIExpression()) #3, !dbg !25
  call void @llvm.dbg.value(metadata i64 2, metadata !18, metadata !DIExpression()) #3, !dbg !13
  call void @llvm.dbg.value(metadata i32 2, metadata !16, metadata !DIExpression()) #3, !dbg !25
  call void @llvm.dbg.value(metadata i64 2, metadata !18, metadata !DIExpression()) #3, !dbg !13
  call void @k(i8* nonnull %0) #3, !dbg !26
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0), !dbg !27
  ret void, !dbg !28
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @k(i8*) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #2

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { argmemonly nounwind }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 7.0.0 (trunk 324259) (llvm/trunk 324261)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "/tmp/test.c", directory: "/")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{i32 7, !"PIC Level", i32 2}
!9 = !{!"clang version 7.0.0 (trunk 324259) (llvm/trunk 324261)"}
!10 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 6, type: !11, isLocal: false, isDefinition: true, scopeLine: 6, isOptimized: true, unit: !0, variables: !2)
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!13 = !DILocation(line: 12, column: 3, scope: !14, inlinedAt: !24)
!14 = distinct !DISubprogram(name: "h", scope: !1, file: !1, line: 10, type: !11, isLocal: false, isDefinition: true, scopeLine: 10, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !15)
!15 = !{!16, !18, !20}
!16 = !DILocalVariable(name: "count", scope: !14, file: !1, line: 11, type: !17)
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!18 = !DILocalVariable(name: "vla_expr", scope: !14, file: !1, line: 12, type: !19)
!19 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!20 = !DILocalVariable(name: "array", scope: !14, file: !1, line: 12, type: !21)
!21 = !DICompositeType(tag: DW_TAG_array_type, baseType: !17, elements: !22)
!22 = !{!23}
!23 = !DISubrange(count: !18)
!24 = distinct !DILocation(line: 7, column: 3, scope: !10)
!25 = !DILocation(line: 11, column: 7, scope: !14, inlinedAt: !24)
!26 = !DILocation(line: 13, column: 3, scope: !14, inlinedAt: !24)
!27 = !DILocation(line: 14, column: 1, scope: !14, inlinedAt: !24)
!28 = !DILocation(line: 8, column: 1, scope: !10)
