; RUN: llc -mtriple=x86_64 -filetype=obj < %s | llvm-dwarfdump -debug-info - | FileCheck %s
;
; CHECK: [[SYM:[a-z0-9]+]]:  DW_TAG_formal_parameter
; CHECK:                     DW_AT_name	("esym")
; CHECK:                     DW_AT_type	([[TYPE:[a-z0-9]+]] "CHARACTER_1")
;
; CHECK:                     DW_TAG_formal_parameter
; CHECK:                       DW_AT_const_value	(7523094288207667809)
; CHECK:                       DW_AT_abstract_origin	([[SYM]] "esym")
;
; CHECK: [[TYPE]]:           DW_TAG_string_type
; CHECK:                       DW_AT_name	("CHARACTER_1")
; CHECK:                       DW_AT_byte_size	(0x08)
;
; The following IR is obtained by compiling the following Fortran
; program with -O2 -g (with irrelevant instructions and metadata
; trimmed):
;
; module semiempirical_corrections
;
;    implicit none
;
; contains
; subroutine gcpcor(n,iz)
;    implicit none
;    integer :: n, i
;    integer :: iz(n)
;    iz(i)=1
;    print*, esym(iz(i))
; end subroutine
;
;
; character*8 function esym(i)
;    integer :: i
;    character*8 elemnt(1)
;    data elemnt/'abcdefgh'/
;    esym=elemnt(i)
;    return
; end function
; end module
;
; The optimizations encode the constant string as an i64 constant in
; the debug info for esym, the return variable of the function esym.

@"semiempirical_corrections_mp_esym_$ELEMNT" = internal unnamed_addr constant [1 x [8 x i8]] [[8 x i8] c"abcdefgh"], align 8, !dbg !0

; Function Attrs: nofree nounwind uwtable
define void @semiempirical_corrections_mp_gcpcor_() local_unnamed_addr #1 !dbg !22 {
alloca_1:
  %"var$47" = alloca i64, align 8, !dbg !34
  call void @llvm.dbg.value(metadata i64 7523094288207667809, metadata !12, metadata !DIExpression()), !dbg !40
  store i64 7523094288207667809, i64* %"var$47", align 8, !dbg !41, !alias.scope !42, !noalias !45
  ret void, !dbg !48
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #3

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #1 = { nofree nounwind uwtable }
attributes #3 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!20, !21}
!llvm.dbg.cu = !{!8}
!omp_offload.info = !{}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "elemnt", linkageName: "semiempirical_corrections_mp_esym_$ELEMNT", scope: !2, file: !3, line: 17, type: !16, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "esym", linkageName: "semiempirical_corrections_mp_esym_", scope: !4, file: !3, line: 15, type: !5, scopeLine: 15, spFlags: DISPFlagDefinition, unit: !8, retainedNodes: !11)
!3 = !DIFile(filename: "se6.f90", directory: "/iusers/cchen15/examples/tests/jr30349/gamess-dga-main/object")
!4 = !DIModule(scope: null, name: "semiempirical_corrections", file: !3, line: 1)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIStringType(name: "CHARACTER_0", size: 64)
!8 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !3, producer: "Intel(R) Fortran 21.0-2745", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !9, globals: !10, splitDebugInlining: false, nameTableKind: None)
!9 = !{}
!10 = !{!0}
!11 = !{!12, !14}
!12 = !DILocalVariable(name: "esym", arg: 1, scope: !2, file: !3, line: 15, type: !13, flags: DIFlagArtificial)
!13 = !DIStringType(name: "CHARACTER_1", size: 64)
!14 = !DILocalVariable(name: "i", arg: 2, scope: !2, file: !3, line: 15, type: !15)
!15 = !DIBasicType(name: "INTEGER*4", size: 32, encoding: DW_ATE_signed)
!16 = !DICompositeType(tag: DW_TAG_array_type, baseType: !17, elements: !18)
!17 = !DIStringType(name: "CHARACTER_2", size: 64)
!18 = !{!19}
!19 = !DISubrange(count: 1, lowerBound: 1)
!20 = !{i32 2, !"Debug Info Version", i32 3}
!21 = !{i32 2, !"Dwarf Version", i32 4}
!22 = distinct !DISubprogram(name: "gcpcor", linkageName: "semiempirical_corrections_mp_gcpcor_", scope: !4, file: !3, line: 6, type: !23, scopeLine: 6, spFlags: DISPFlagDefinition, unit: !8)
!23 = !DISubroutineType(types: !24)
!24 = !{null}
!34 = !DILocation(line: 6, column: 19, scope: !22)
!38 = !DILocation(line: 15, column: 27, scope: !2, inlinedAt: !39)
!39 = distinct !DILocation(line: 11, column: 13, scope: !22)
!40 = !DILocation(line: 0, scope: !2, inlinedAt: !39)
!41 = !DILocation(line: 19, column: 5, scope: !2, inlinedAt: !39)
!42 = !{!43}
!43 = distinct !{!43, !44, !"semiempirical_corrections_mp_esym_: %semiempirical_corrections_mp_esym_$ESYM"}
!44 = distinct !{!44, !"semiempirical_corrections_mp_esym_"}
!45 = !{!46}
!46 = distinct !{!46, !44, !"semiempirical_corrections_mp_esym_: %I"}
!48 = !DILocation(line: 12, column: 1, scope: !22)
