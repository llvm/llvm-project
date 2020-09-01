;; Test for !DIStringType.!DIStringType is used to construct a Fortran
;; CHARACTER intrinsic type, with a LEN type parameter where LEN is a
;; dynamic parameter as in a deferred-length CHARACTER. LLVM after
;; processing this !DIStringType metadata, generates DW_AT_string_length attribute.
;; !DIStringType(name: "character(*)", stringLength: !{{[0-9]+}},
;;		stringLengthExpression: !DIExpression(), size: 32)

; RUN: %llc_dwarf -filetype=obj  %s -o - | llvm-dwarfdump - | FileCheck %s
; CHECK:       DW_TAG_string_type
; CHECK:                          DW_AT_name  ("character(*)!2")
; CHECK-NEXT:                     DW_AT_string_length

;; sample fortran testcase involving assumed length string type.
;; program assumedLength
;;   call sub('Hello')
;;   call sub('Goodbye')
;;   contains
;;   subroutine sub(string)
;;           implicit none
;;           character(len=*), intent(in) :: string
;;           print *, string
;;   end subroutine sub
;; end program assumedLength

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.struct_ul_MAIN__324 = type <{ i8* }>

@.C327_MAIN_ = internal constant [7 x i8] c"Goodbye"
@.C326_MAIN_ = internal constant [5 x i8] c"Hello"
@.C306_MAIN_ = internal constant i32 0
@.C336_assumedlength_sub = internal constant i32 14
@.C306_assumedlength_sub = internal constant i32 0
@.C307_assumedlength_sub = internal constant i64 0
@.C331_assumedlength_sub = internal constant i32 6
@.C329_assumedlength_sub = internal constant [8 x i8] c"test.f90"
@.C328_assumedlength_sub = internal constant i32 8

define void @MAIN_() !dbg !5 {
L.entry:
  %.S0000_331 = alloca %struct.struct_ul_MAIN__324, align 8
  %0 = bitcast i32* @.C306_MAIN_ to i8*, !dbg !8
  %1 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !8
  call void (i8*, ...) %1(i8* %0), !dbg !8
  br label %L.LB1_335

L.LB1_335:                                        ; preds = %L.entry
  %2 = bitcast [5 x i8]* @.C326_MAIN_ to i64*, !dbg !9
  %3 = bitcast %struct.struct_ul_MAIN__324* %.S0000_331 to i64*, !dbg !9
  call void @assumedlength_sub(i64* %2, i64 5, i64* %3), !dbg !9
  %4 = bitcast [7 x i8]* @.C327_MAIN_ to i64*, !dbg !10
  %5 = bitcast %struct.struct_ul_MAIN__324* %.S0000_331 to i64*, !dbg !10
  call void @assumedlength_sub(i64* %4, i64 7, i64* %5), !dbg !10
  ret void, !dbg !11
}

define internal void @assumedlength_sub(i64* noalias %string, i64 %.U0001.arg, i64* noalias %.S0000) !dbg !12 {
L.entry:
  %.U0001.addr = alloca i64, align 8
  %z__io_333 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i64* %string, metadata !16, metadata !DIExpression()), !dbg !20
  store i64 %.U0001.arg, i64* %.U0001.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %.U0001.addr, metadata !18, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.declare(metadata i64* %.S0000, metadata !21, metadata !DIExpression()), !dbg !20
  br label %L.LB2_347

L.LB2_347:                                        ; preds = %L.entry
  %0 = bitcast i32* @.C328_assumedlength_sub to i8*, !dbg !23
  %1 = bitcast [8 x i8]* @.C329_assumedlength_sub to i8*, !dbg !23
  %2 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !23
  call void (i8*, i8*, i64, ...) %2(i8* %0, i8* %1, i64 8), !dbg !23
  %3 = bitcast i32* @.C331_assumedlength_sub to i8*, !dbg !23
  %4 = bitcast i32* @.C306_assumedlength_sub to i8*, !dbg !23
  %5 = bitcast i32* @.C306_assumedlength_sub to i8*, !dbg !23
  %6 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !23
  %7 = call i32 (i8*, i8*, i8*, i8*, ...) %6(i8* %3, i8* null, i8* %4, i8* %5), !dbg !23
  call void @llvm.dbg.declare(metadata i32* %z__io_333, metadata !24, metadata !DIExpression()), !dbg !20
  store i32 %7, i32* %z__io_333, align 4, !dbg !23
  %8 = bitcast i64* %string to i8*, !dbg !23
  %9 = load i64, i64* %.U0001.addr, align 8, !dbg !23
  %10 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !23
  %11 = call i32 (i8*, i32, i64, ...) %10(i8* %8, i32 14, i64 %9), !dbg !23
  store i32 %11, i32* %z__io_333, align 4, !dbg !23
  %12 = call i32 (...) @f90io_ldw_end(), !dbg !23
  store i32 %12, i32* %z__io_333, align 4, !dbg !23
  ret void, !dbg !26
}

declare signext i32 @f90io_ldw_end(...)

declare signext i32 @f90io_sc_ch_ldw(...)

declare signext i32 @f90io_print_init(...)

declare void @f90io_src_info03a(...)

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata)

declare void @fort_init(...)

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "test.f90", directory: "/tmp")
!4 = !{}
!5 = distinct !DISubprogram(name: "assumedlength", scope: !2, file: !3, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocation(line: 1, column: 1, scope: !5)
!9 = !DILocation(line: 2, column: 1, scope: !5)
!10 = !DILocation(line: 3, column: 1, scope: !5)
!11 = !DILocation(line: 4, column: 1, scope: !5)
!12 = distinct !DISubprogram(name: "sub", scope: !5, file: !3, line: 5, type: !13, scopeLine: 5, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!13 = !DISubroutineType(types: !14)
!14 = !{null, !15}
!15 = !DIStringType(name: "character(*)!1", size: 32)
!16 = !DILocalVariable(name: "string", arg: 1, scope: !12, file: !3, type: !17)
!17 = !DIStringType(name: "character(*)!2", stringLength: !18, stringLengthExpression: !DIExpression(), size: 32)
!18 = !DILocalVariable(arg: 2, scope: !12, file: !3, type: !19, flags: DIFlagArtificial)
!19 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!20 = !DILocation(line: 0, scope: !12)
!21 = !DILocalVariable(arg: 3, scope: !12, file: !3, type: !22, flags: DIFlagArtificial)
!22 = !DIBasicType(name: "uinteger*8", size: 64, align: 64, encoding: DW_ATE_unsigned)
!23 = !DILocation(line: 8, column: 1, scope: !12)
!24 = distinct !DILocalVariable(scope: !12, file: !3, type: !25, flags: DIFlagArtificial)
!25 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!26 = !DILocation(line: 9, column: 1, scope: !12)
