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
;         complex*8 cmp8
;         complex*16 cmp16
;         complex*32 cmp32
;
;         common /com/ d, l, c
;
;         array1(1) = 1
;         array2(3) = 2
;         d = 8.0
;         l = .TRUE.
;         c = 'oooooo'
;         cmp8 = (8.8, 1.1)
;         cmp16 = (16.16, 2.2)
;         cmp32 = (32.32, 3.3)
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
; CHECK:      Type: _Complex __float128 (0x53)
; CHECK-NEXT: Flags [
; CHECK-NEXT: ]
; CHECK-NEXT: VarName: CMP32
;
; CHECK:      Type: _Complex double (0x51)
; CHECK-NEXT: Flags [
; CHECK-NEXT: ]
; CHECK-NEXT: VarName: CMP16
;
; CHECK:      Type: _Complex float (0x50)
; CHECK-NEXT: Flags [
; CHECK-NEXT: ]
; CHECK-NEXT: VarName: CMP8
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

%complex_256bit = type { fp128, fp128 }
%complex_128bit = type { double, double }
%complex_64bit = type { float, float }

@strlit = internal unnamed_addr constant [6 x i8] c"oooooo"
@COM = common unnamed_addr global [18 x i8] zeroinitializer, align 32, !dbg !0, !dbg !9, !dbg !12
@"ARRAY$ARRAY2" = internal global [8 x i32] zeroinitializer, align 16, !dbg !15
@"ARRAY$ARRAY1" = internal global [10 x i32] zeroinitializer, align 16, !dbg !21
@0 = internal unnamed_addr constant i32 2

; Function Attrs: noinline nounwind optnone uwtable
define void @MAIN__() #0 !dbg !3 {
alloca_0:
  %"$io_ctx" = alloca [6 x i64], align 8
  %"ARRAY$CMP32" = alloca %complex_256bit, align 16
  %"ARRAY$CMP16" = alloca %complex_128bit, align 8, !dbg !42
  %"ARRAY$CMP8" = alloca %complex_64bit, align 8, !dbg !42
  call void @llvm.dbg.declare(metadata ptr %"ARRAY$CMP32", metadata !27, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.declare(metadata ptr %"ARRAY$CMP16", metadata !29, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.declare(metadata ptr %"ARRAY$CMP8", metadata !31, metadata !DIExpression()), !dbg !40
  %strlit_fetch.1 = load [6 x i8], ptr @strlit, align 1, !dbg !39
  %func_result = call i32 @for_set_reentrancy(ptr @0), !dbg !39
  store i32 1, ptr @"ARRAY$ARRAY1", align 1, !dbg !43
  store i32 2, ptr @"ARRAY$ARRAY2", align 1, !dbg !44
  store double 8.000000e+00, ptr @COM, align 1, !dbg !45
  store i32 -1, ptr getelementptr inbounds ([18 x i8], ptr @COM, i32 0, i64 8), align 1, !dbg !46
  call void @llvm.for.cpystr.i64.i64.i64(ptr getelementptr inbounds ([18 x i8], ptr @COM, i32 0, i64 12), i64 6, ptr @strlit, i64 3, i64 0, i1 false), !dbg !47
  store %complex_64bit { float 0x40219999A0000000, float 0x3FF19999A0000000 }, ptr %"ARRAY$CMP8", align 8, !dbg !48
  store %complex_128bit { double 0x403028F5C0000000, double 0x40019999A0000000 }, ptr %"ARRAY$CMP16", align 8, !dbg !49
  store %complex_256bit { fp128 0xL00000000000000004004028F5C000000, fp128 0xL00000000000000004000A66666000000 }, ptr %"ARRAY$CMP32", align 16, !dbg !50
  ret void, !dbg !51
}

declare i32 @for_set_reentrancy(ptr nocapture readonly)

; Function Attrs: nounwind readnone speculatable
declare ptr @llvm.intel.subscript.p0.i64.i64.p0.i64(i8, i64, i64, ptr, i64) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.for.cpystr.i64.i64.i64(ptr noalias nocapture writeonly, i64, ptr noalias nocapture readonly, i64, i64, i1 immarg) #2

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #3

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="none" "intel-lang"="fortran" "min-legal-vector-width"="0" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { argmemonly nofree nosync nounwind willreturn }
attributes #3 = { nocallback nofree nosync nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!34, !35, !36}
!llvm.dbg.cu = !{!7}
!omp_offload.info = !{}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "D", linkageName: "COM", scope: !2, file: !4, line: 5, type: !33, isLocal: false, isDefinition: true)
!2 = !DICommonBlock(scope: !3, declaration: null, name: "COM", file: !4, line: 8)
!3 = distinct !DISubprogram(name: "ARRAY", linkageName: "MAIN__", scope: !4, file: !4, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !7, retainedNodes: !26)
!4 = !DIFile(filename: "fortran-basic.f", directory: "d:\\temp")
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !4, producer: "Fortran", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !8, splitDebugInlining: false, nameTableKind: None)
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
!26 = !{!27, !29, !31}
!27 = !DILocalVariable(name: "CMP32", scope: !3, file: !4, line: 10, type: !28)
!28 = !DIBasicType(name: "COMPLEX*32", size: 256, encoding: DW_ATE_complex_float)
!29 = !DILocalVariable(name: "CMP16", scope: !3, file: !4, line: 9, type: !30)
!30 = !DIBasicType(name: "COMPLEX*16", size: 128, encoding: DW_ATE_complex_float)
!31 = !DILocalVariable(name: "CMP8", scope: !3, file: !4, line: 8, type: !32)
!32 = !DIBasicType(name: "COMPLEX*8", size: 64, encoding: DW_ATE_complex_float)
!33 = !DIBasicType(name: "REAL*8", size: 64, encoding: DW_ATE_float)
!34 = !{i32 7, !"PIC Level", i32 2}
!35 = !{i32 2, !"Debug Info Version", i32 3}
!36 = !{i32 2, !"CodeView", i32 1}
!39 = !DILocation(line: 1, column: 10, scope: !3)
!40 = !DILocation(line: 8, column: 9, scope: !3)
!41 = !DILocation(line: 9, column: 9, scope: !3)
!42 = !DILocation(line: 10, column: 9, scope: !3)
!43 = !DILocation(line: 14, column: 9, scope: !3)
!44 = !DILocation(line: 15, column: 9, scope: !3)
!45 = !DILocation(line: 16, column: 9, scope: !3)
!46 = !DILocation(line: 17, column: 9, scope: !3)
!47 = !DILocation(line: 18, column: 9, scope: !3)
!48 = !DILocation(line: 19, column: 9, scope: !3)
!49 = !DILocation(line: 20, column: 9, scope: !3)
!50 = !DILocation(line: 21, column: 9, scope: !3)
!51 = !DILocation(line: 22, column: 2, scope: !3)
