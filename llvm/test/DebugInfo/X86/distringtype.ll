;; Test for !DIStringType.!DIStringType is used to construct a Fortran
;; CHARACTER intrinsic type, with a LEN type parameter where LEN is a
;; dynamic parameter as in a deferred-length CHARACTER. LLVM after
;; processing !DIStringType metadata in either of the following forms,
;; generates DW_AT_string_length attribute
;; !DIStringType(name: "character(*)", stringLength: !{{[0-9]+}})
;; !DIStringType(name: "character(*)", stringLengthExpr: !DIExpression(...))
;;
;; !DIStringType has an optional stringLocationExpr field. This
;; tests also verifies that field gets emitted as DW_AT_data_location
;; in the DIE.

; RUN: llc -filetype=obj  %s -o - | llvm-dwarfdump - | FileCheck %s
; CHECK:       DW_TAG_string_type
; CHECK:                          DW_AT_name  (".str.DEFERRED")
; CHECK-NEXT:                     DW_AT_string_length (DW_OP_push_object_address, DW_OP_plus_uconst 0x8)
; CHECK-NEXT:                     DW_AT_data_location (DW_OP_push_object_address, DW_OP_deref)
; CHECK:       DW_TAG_string_type
; CHECK:                          DW_AT_name  ("character(*)!2")
; CHECK-NEXT:                     DW_AT_string_length

;; sample fortran testcase involving deferred and assumed length string types.
;; program assumedLength
;;   character(len=:), allocatable :: deferred
;;   allocate(character(10)::deferred)
;;   call sub('Hello')
;;   call sub('Goodbye')
;;   contains
;;   subroutine sub(string)
;;           implicit none
;;           character(len=*), intent(in) :: string
;;           print *, string
;;   end subroutine sub
;; end program assumedLength

; ModuleID = 'distring.f90'
source_filename = "distring.f90"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"QNCA_a0$i8*$rank0$" = type { i8*, i64, i64, i64, i64, i64 }

@0 = internal unnamed_addr constant [7 x i8] c"Goodbye"
@1 = internal unnamed_addr constant [5 x i8] c"Hello"
@"assumedlength_$DEFERRED" = internal global %"QNCA_a0$i8*$rank0$" zeroinitializer, !dbg !0
@2 = internal unnamed_addr constant i32 2

; Function Attrs: noinline nounwind uwtable
define void @MAIN__() #0 !dbg !2 {
alloca_0:
  %"var$1" = alloca [8 x i64], align 8
  %"var$2" = alloca i64, align 8
  %strlit = load [7 x i8], [7 x i8]* @0, align 1
  %strlit1 = load [5 x i8], [5 x i8]* @1, align 1
  %func_result = call i32 @for_set_reentrancy(i32* @2), !dbg !12
  %val_fetch = load i64, i64* getelementptr inbounds (%"QNCA_a0$i8*$rank0$", %"QNCA_a0$i8*$rank0$"* @"assumedlength_$DEFERRED", i32 0, i32 1), align 1, !dbg !13
  %val_fetch4 = load i64, i64* getelementptr inbounds (%"QNCA_a0$i8*$rank0$", %"QNCA_a0$i8*$rank0$"* @"assumedlength_$DEFERRED", i32 0, i32 3), align 1, !dbg !13
  %and = and i64 %val_fetch4, 256, !dbg !13
  %lshr = lshr i64 %and, 8, !dbg !13
  %shl = shl i64 %lshr, 8, !dbg !13
  %or = or i64 133, %shl, !dbg !13
  %and5 = and i64 %val_fetch4, 1030792151040, !dbg !13
  %lshr6 = lshr i64 %and5, 36, !dbg !13
  %and7 = and i64 %or, -1030792151041, !dbg !13
  %shl8 = shl i64 %lshr6, 36, !dbg !13
  %or9 = or i64 %and7, %shl8, !dbg !13
  store i64 %or9, i64* getelementptr inbounds (%"QNCA_a0$i8*$rank0$", %"QNCA_a0$i8*$rank0$"* @"assumedlength_$DEFERRED", i32 0, i32 3), align 1, !dbg !13
  store i64 10, i64* getelementptr inbounds (%"QNCA_a0$i8*$rank0$", %"QNCA_a0$i8*$rank0$"* @"assumedlength_$DEFERRED", i32 0, i32 1), align 1, !dbg !13
  store i64 0, i64* getelementptr inbounds (%"QNCA_a0$i8*$rank0$", %"QNCA_a0$i8*$rank0$"* @"assumedlength_$DEFERRED", i32 0, i32 4), align 1, !dbg !13
  store i64 0, i64* getelementptr inbounds (%"QNCA_a0$i8*$rank0$", %"QNCA_a0$i8*$rank0$"* @"assumedlength_$DEFERRED", i32 0, i32 2), align 1, !dbg !13
  %val_fetch10 = load i64, i64* getelementptr inbounds (%"QNCA_a0$i8*$rank0$", %"QNCA_a0$i8*$rank0$"* @"assumedlength_$DEFERRED", i32 0, i32 3), align 1, !dbg !13
  %val_fetch11 = load i64, i64* getelementptr inbounds (%"QNCA_a0$i8*$rank0$", %"QNCA_a0$i8*$rank0$"* @"assumedlength_$DEFERRED", i32 0, i32 3), align 1, !dbg !13
  %and12 = and i64 %val_fetch11, -68451041281, !dbg !13
  %or13 = or i64 %and12, 1073741824, !dbg !13
  store i64 %or13, i64* getelementptr inbounds (%"QNCA_a0$i8*$rank0$", %"QNCA_a0$i8*$rank0$"* @"assumedlength_$DEFERRED", i32 0, i32 3), align 1, !dbg !13
  %and14 = and i64 %val_fetch10, 1, !dbg !13
  %shl15 = shl i64 %and14, 1, !dbg !13
  %int_zext = trunc i64 %shl15 to i32, !dbg !13
  %or16 = or i32 0, %int_zext, !dbg !13
  %and17 = and i32 %or16, -17, !dbg !13
  %and18 = and i64 %val_fetch10, 256, !dbg !13
  %lshr19 = lshr i64 %and18, 8, !dbg !13
  %and20 = and i32 %and17, -2097153, !dbg !13
  %shl21 = shl i64 %lshr19, 21, !dbg !13
  %int_zext22 = trunc i64 %shl21 to i32, !dbg !13
  %or23 = or i32 %and20, %int_zext22, !dbg !13
  %and24 = and i64 %val_fetch10, 1030792151040, !dbg !13
  %lshr25 = lshr i64 %and24, 36, !dbg !13
  %and26 = and i32 %or23, -31457281, !dbg !13
  %shl27 = shl i64 %lshr25, 21, !dbg !13
  %int_zext28 = trunc i64 %shl27 to i32, !dbg !13
  %or29 = or i32 %and26, %int_zext28, !dbg !13
  %and30 = and i64 %val_fetch10, 1099511627776, !dbg !13
  %lshr31 = lshr i64 %and30, 40, !dbg !13
  %and32 = and i32 %or29, -33554433, !dbg !13
  %shl33 = shl i64 %lshr31, 25, !dbg !13
  %int_zext34 = trunc i64 %shl33 to i32, !dbg !13
  %or35 = or i32 %and32, %int_zext34, !dbg !13
  %and36 = and i32 %or35, -2031617, !dbg !13
  %or37 = or i32 %and36, 262144, !dbg !13
  %func_result3 = call i32 @for_alloc_allocatable(i64 10, i8** getelementptr inbounds (%"QNCA_a0$i8*$rank0$", %"QNCA_a0$i8*$rank0$"* @"assumedlength_$DEFERRED", i32 0, i32 0), i32 %or37), !dbg !13
  call void @assumedlength_IP_sub_(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @1, i32 0, i32 0), i64 5), !dbg !14
  call void @assumedlength_IP_sub_(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @0, i32 0, i32 0), i64 7), !dbg !15
  ret void, !dbg !16
}

; Function Attrs: noinline nounwind uwtable
define void @assumedlength_IP_sub_(i8* noalias nocapture readonly %STRING, i64 %"STRING.len$val") #0 !dbg !17 {
alloca_1:
  %"var$3" = alloca [8 x i64], align 8
  %STRING.len = alloca i64, align 8, !dbg !23
  %"var$4" = alloca i32, align 4, !dbg !23
  %"(&)val$" = alloca [4 x i8], align 1, !dbg !23
  %argblock = alloca { i64, i8* }, align 8, !dbg !23
  call void @llvm.dbg.declare(metadata i64* %STRING.len, metadata !19, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.declare(metadata i8* %STRING, metadata !21, metadata !DIExpression()), !dbg !24
  store i64 %"STRING.len$val", i64* %STRING.len, align 8
  %strlit = load [7 x i8], [7 x i8]* @0, align 1, !dbg !25
  %strlit1 = load [5 x i8], [5 x i8]* @1, align 1, !dbg !25
  %STRING.len_fetch = load i64, i64* %STRING.len, align 1, !dbg !26
  store [4 x i8] c"8\04\01\00", [4 x i8]* %"(&)val$", align 1, !dbg !26
  %BLKFIELD_ = getelementptr inbounds { i64, i8* }, { i64, i8* }* %argblock, i32 0, i32 0, !dbg !26
  store i64 %STRING.len_fetch, i64* %BLKFIELD_, align 1, !dbg !26
  %BLKFIELD_3 = getelementptr inbounds { i64, i8* }, { i64, i8* }* %argblock, i32 0, i32 1, !dbg !26
  store i8* %STRING, i8** %BLKFIELD_3, align 1, !dbg !26
  %"(i8*)var$3$" = bitcast [8 x i64]* %"var$3" to i8*, !dbg !26
  %"(i8*)(&)val$$" = bitcast [4 x i8]* %"(&)val$" to i8*, !dbg !26
  %"(i8*)argblock$" = bitcast { i64, i8* }* %argblock to i8*, !dbg !26
  %func_result = call i32 (i8*, i32, i64, i8*, i8*, ...) @for_write_seq_lis(i8* %"(i8*)var$3$", i32 -1, i64 1239157112576, i8* %"(i8*)(&)val$$", i8* %"(i8*)argblock$"), !dbg !26
  ret void, !dbg !27
}

declare i32 @for_set_reentrancy(i32*)

declare i32 @for_alloc_allocatable(i64, i8**, i32)

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i32 @for_write_seq_lis(i8*, i32, i64, i8*, i8*, ...)

attributes #0 = { noinline nounwind uwtable "intel-lang"="fortran" "min-legal-vector-width"="0" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!10, !11}
!llvm.dbg.cu = !{!6}
!omp_offload.info = !{}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "deferred", linkageName: "assumedlength_$DEFERRED", scope: !2, file: !3, line: 2, type: !9, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "assumedlength", linkageName: "MAIN__", scope: !3, file: !3, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !7)
!3 = !DIFile(filename: "distring.f90", directory: "/iusers/cchen15/examples/tests")
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !3, producer: "Intel(R) Fortran 21.0-2142", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !7, globals: !8, splitDebugInlining: false, nameTableKind: None)
!7 = !{}
!8 = !{!0}
!9 = !DIStringType(name: ".str.DEFERRED", stringLengthExpression: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 8), stringLocationExpression: !DIExpression(DW_OP_push_object_address, DW_OP_deref))
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !DILocation(line: 1, column: 9, scope: !2)
!13 = !DILocation(line: 3, column: 3, scope: !2)
!14 = !DILocation(line: 4, column: 8, scope: !2)
!15 = !DILocation(line: 5, column: 8, scope: !2)
!16 = !DILocation(line: 6, column: 3, scope: !2)
!17 = distinct !DISubprogram(name: "sub", linkageName: "assumedlength_IP_sub_", scope: !3, file: !3, line: 7, type: !4, scopeLine: 7, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !18)
!18 = !{!19, !21}
!19 = !DILocalVariable(name: "STRING.len", scope: !17, type: !20, flags: DIFlagArtificial)
!20 = !DIBasicType(name: "INTEGER*8", size: 64, encoding: DW_ATE_signed)
!21 = !DILocalVariable(name: "string", arg: 1, scope: !17, file: !3, line: 7, type: !22)
!22 = !DIStringType(name: "character(*)!2", stringLength: !19)
!23 = !DILocation(line: 0, scope: !17)
!24 = !DILocation(line: 7, column: 18, scope: !17)
!25 = !DILocation(line: 7, column: 14, scope: !17)
!26 = !DILocation(line: 10, column: 11, scope: !17)
!27 = !DILocation(line: 11, column: 3, scope: !17)
