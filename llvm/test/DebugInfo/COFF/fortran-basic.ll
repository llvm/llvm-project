; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s
;
; The IR in this test derives from the following Fortran program:
;	program array
;	  integer array1, array2
;	  dimension array1(10)
;         dimension array2(3:10)
;         double precision d
;         logical l
;         character*6 c
;
;         common /com/ d, l, c
;
;         array1(1) = 1
;         array2(3) = 2
;         d = 8.0
;         l = .TRUE.
;         c = 'oooooo'
;	end
;
; CHECK: Array ([[array2_t:.*]]) {
; CHECK-NEXT: TypeLeafKind: LF_ARRAY
; CHECK-NEXT: ElementType: int
; CHECK-NEXT: IndexType: unsigned __int64
; CHECK-NEXT: SizeOf: 32
;
; CHECK: Array ([[array1_t:.*]]) {
; CHECK-NEXT: TypeLeafKind: LF_ARRAY
; CHECK-NEXT: ElementType: int
; CHECK-NEXT: IndexType: unsigned __int64
; CHECK-NEXT: SizeOf: 40
;
; CHECK: Array ([[char_6_t:.*]]) {
; CHECK-NEXT: TypeLeafKind: LF_ARRAY
; CHECK-NEXT: ElementType: char
; CHECK-NEXT: IndexType: unsigned __int64
; CHECK-NEXT: SizeOf: 6
; CHECK-NEXT: CHARACTER_0
;
; CHECK: DataOffset: ARRAY$ARRAY2+0x0
; CHECK-NEXT: Type: [[array2_t]]
; CHECK-NEXT: DisplayName: ARRAY2
; CHECK-NEXT: LinkageName: ARRAY$ARRAY2
;
; CHECK: DataOffset: ARRAY$ARRAY1+0x0
; CHECK-NEXT: Type: [[array1_t]]
; CHECK-NEXT: DisplayName: ARRAY1
; CHECK-NEXT: LinkageName: ARRAY$ARRAY1
;
; CHECK: DataOffset: COM+0x0
; CHECK-NEXT: Type: double
; CHECK-NEXT: DisplayName: D
; CHECK-NEXT: LinkageName: COM
;
; CHECK: DataOffset: COM+0x8
; CHECK-NEXT: Type: __bool32
; CHECK-NEXT: DisplayName: L
; CHECK-NEXT: LinkageName: COM
;
; CHECK: DataOffset: COM+0xC
; CHECK-NEXT: Type: CHARACTER_0 ([[char_6_t]])
; CHECK-NEXT: DisplayName: C
; CHECK-NEXT: LinkageName: COM

; ModuleID = 'fortran-basic.f'
source_filename = "fortran-basic.f"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

@strlit = internal unnamed_addr constant [6 x i8] c"oooooo"
@COM = common unnamed_addr global [18 x i8] zeroinitializer, align 32, !dbg !0, !dbg !9, !dbg !12
@"ARRAY$ARRAY2" = internal global [8 x i32] zeroinitializer, align 16, !dbg !15
@"ARRAY$ARRAY1" = internal global [10 x i32] zeroinitializer, align 16, !dbg !21
@0 = internal unnamed_addr constant i32 2

; Function Attrs: noinline nounwind optnone uwtable
define void @MAIN__() #0 !dbg !3 {
alloca_0:
  %"$io_ctx" = alloca [6 x i64], align 8
  %strlit_fetch.1 = load [6 x i8], [6 x i8]* @strlit, align 1, !dbg !39
  %func_result = call i32 @for_set_reentrancy(i32* @0), !dbg !39
  store i32 1, i32* getelementptr inbounds ([10 x i32], [10 x i32]* @"ARRAY$ARRAY1", i32 0, i32 0), align 1, !dbg !40
  store i32 2, i32* getelementptr inbounds ([8 x i32], [8 x i32]* @"ARRAY$ARRAY2", i32 0, i32 0), align 1, !dbg !41
  store double 8.000000e+00, double* bitcast ([18 x i8]* @COM to double*), align 1, !dbg !42
  store i32 -1, i32* bitcast (i8* getelementptr inbounds ([18 x i8], [18 x i8]* @COM, i32 0, i64 8) to i32*), align 1, !dbg !43
  call void @llvm.for.cpystr.i64.i64.i64(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @COM, i32 0, i64 12), i64 6, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @strlit, i32 0, i32 0), i64 3, i64 0, i1 false), !dbg !44
  ret void, !dbg !45
}

declare i32 @for_set_reentrancy(i32* nocapture readonly)

; Function Attrs: nounwind readnone speculatable
declare i32* @llvm.intel.subscript.p0i32.i64.i64.p0i32.i64(i8, i64, i64, i32*, i64) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.for.cpystr.i64.i64.i64(i8* noalias nocapture writeonly, i64, i8* noalias nocapture readonly, i64, i64, i1 immarg) #2

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="none" "intel-lang"="fortran" "min-legal-vector-width"="0" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { argmemonly nofree nosync nounwind willreturn }

!llvm.module.flags = !{!28, !29, !30}
!llvm.dbg.cu = !{!7}
!omp_offload.info = !{}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "D", linkageName: "COM", scope: !2, file: !4, line: 5, type: !27, isLocal: false, isDefinition: true)
!2 = !DICommonBlock(scope: !3, declaration: null, name: "COM", file: !4, line: 8)
!3 = distinct !DISubprogram(name: "ARRAY", linkageName: "MAIN__", scope: !4, file: !4, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !7, retainedNodes: !26)
!4 = !DIFile(filename: "fortran-basic.f", directory: "d:\\iusers\\cchen15\\examples\\tests\\vsdF-nightly\\vsdF\\opt_none_debug")
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !4, producer: "Intel(R) Fortran 22.0-1034", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !8, splitDebugInlining: false, nameTableKind: None)
!8 = !{!0, !9, !12, !15, !21}
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression(DW_OP_plus_uconst, 8))
!10 = distinct !DIGlobalVariable(name: "L", linkageName: "COM", scope: !2, file: !4, line: 6, type: !11, isLocal: false, isDefinition: true)
!11 = !DIBasicType(name: "LOGICAL*4", size: 32, encoding: DW_ATE_boolean)
!12 = !DIGlobalVariableExpression(var: !13, expr: !DIExpression(DW_OP_plus_uconst, 12))
!13 = distinct !DIGlobalVariable(name: "C", linkageName: "COM", scope: !2, file: !4, line: 7, type: !14, isLocal: false, isDefinition: true)
!14 = !DIStringType(name: "CHARACTER_0", size: 48)
!15 = !DIGlobalVariableExpression(var: !16, expr: !DIExpression())
!16 = distinct !DIGlobalVariable(name: "ARRAY2", linkageName: "ARRAY$ARRAY2", scope: !3, file: !4, line: 2, type: !17, isLocal: true, isDefinition: true)
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, elements: !19)
!18 = !DIBasicType(name: "INTEGER*4", size: 32, encoding: DW_ATE_signed)
!19 = !{!20}
!20 = !DISubrange(lowerBound: 3, upperBound: 10)
!21 = !DIGlobalVariableExpression(var: !22, expr: !DIExpression())
!22 = distinct !DIGlobalVariable(name: "ARRAY1", linkageName: "ARRAY$ARRAY1", scope: !3, file: !4, line: 2, type: !23, isLocal: true, isDefinition: true)
!23 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, elements: !24)
!24 = !{!25}
!25 = !DISubrange(count: 10, lowerBound: 1)
!26 = !{}
!27 = !DIBasicType(name: "REAL*8", size: 64, encoding: DW_ATE_float)
!28 = !{i32 7, !"PIC Level", i32 2}
!29 = !{i32 2, !"Debug Info Version", i32 3}
!30 = !{i32 2, !"CodeView", i32 1}
!39 = !DILocation(line: 1, column: 10, scope: !3)
!40 = !DILocation(line: 9, column: 9, scope: !3)
!41 = !DILocation(line: 10, column: 9, scope: !3)
!42 = !DILocation(line: 11, column: 9, scope: !3)
!43 = !DILocation(line: 12, column: 9, scope: !3)
!44 = !DILocation(line: 13, column: 9, scope: !3)
!45 = !DILocation(line: 14, column: 2, scope: !3)
