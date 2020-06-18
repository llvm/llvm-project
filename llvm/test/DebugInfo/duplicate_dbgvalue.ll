; RUN: opt -instcombine -S -o - < %s | FileCheck %s

; CHECK-LABEL: %4 = load i32, i32* %i1_311
; CHECK: call void @llvm.dbg.value(metadata i32 %4
; Next instruction should not be duplicate dbg.value intrinsic.
; CHECK-NEXT: @f90io_sc_i_ldw

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;program main
;integer :: res
; res = mfun()
; print *, res
;contains
; function mfun()
;  integer :: i1
;  i1 = 5
;  mfun = fun(i1)
;  write (*,*) i1
; end function
; function fun(a)
;    integer, intent (in) :: a
;    fun = a
; end function
;end program main
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; ModuleID = 'duplicate_dbgvalue.ll'
source_filename = "duplicate_dbgvalue.ll"
target datalayout = "e-p:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.BSS1 = type <{ [4 x i8] }>

@.BSS1 = internal unnamed_addr global %struct.BSS1 zeroinitializer, align 32, !dbg !0
@.C303_MAIN_ = internal constant i32 6
@.C300_MAIN_ = internal constant [22 x i8] c"duplicate_dbgvalue.f90"
@.C302_MAIN_ = internal constant i32 4
@.C283_MAIN_ = internal constant i32 0
@.C283_main_mfun = internal constant i32 0
@.C302_main_mfun = internal constant i32 6
@.C300_main_mfun = internal constant [22 x i8] c"duplicate_dbgvalue.f90"
@.C313_main_mfun = internal constant i32 10

define void @MAIN_() local_unnamed_addr !dbg !2 {
L.entry:
  call void (i8*, ...) bitcast (void (...)* @fort_init to void (i8*, ...)*)(i8* bitcast (i32* @.C283_MAIN_ to i8*)), !dbg !16
  %0 = call fastcc i32 @main_mfun(), !dbg !18
  store i32 %0, i32* bitcast (%struct.BSS1* @.BSS1 to i32*), align 32, !dbg !18
  call void (i8*, i8*, i64, ...) bitcast (void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*)(i8* bitcast (i32* @.C302_MAIN_ to i8*), i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.C300_MAIN_, i64 0, i64 0), i64 22), !dbg !23
  %1 = call i32 (i8*, i8*, i8*, i8*, ...) bitcast (i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*)(i8* bitcast (i32* @.C303_MAIN_ to i8*), i8* null, i8* bitcast (i32* @.C283_MAIN_ to i8*), i8* bitcast (i32* @.C283_MAIN_ to i8*)), !dbg !23
  call void @llvm.dbg.value(metadata i32 %1, metadata !24, metadata !DIExpression()), !dbg !25
  %2 = load i32, i32* bitcast (%struct.BSS1* @.BSS1 to i32*), align 32, !dbg !23
  %3 = call i32 (i32, i32, ...) bitcast (i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*)(i32 %2, i32 25), !dbg !23
  call void @llvm.dbg.value(metadata i32 %3, metadata !24, metadata !DIExpression()), !dbg !25
  %4 = call i32 (...) @f90io_ldw_end(), !dbg !23
  call void @llvm.dbg.value(metadata i32 %4, metadata !24, metadata !DIExpression()), !dbg !25
  ret void, !dbg !26
}

define internal fastcc signext i32 @main_mfun() unnamed_addr !dbg !27 {
L.entry:
  %i1_311 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i64* undef, metadata !31, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.declare(metadata i32* %i1_311, metadata !35, metadata !DIExpression()), !dbg !33
  store i32 5, i32* %i1_311, align 4, !dbg !36
  %0 = bitcast i32* %i1_311 to i64*, !dbg !41
  %1 = call fastcc float @main_fun(i64* %0), !dbg !41
  %2 = fptosi float %1 to i32, !dbg !41
  call void (i8*, i8*, i64, ...) bitcast (void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*)(i8* bitcast (i32* @.C313_main_mfun to i8*), i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.C300_main_mfun, i32 0, i32 0), i64 22), !dbg !42
  %3 = call i32 (i8*, i8*, i8*, i8*, ...) bitcast (i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*)(i8* bitcast (i32* @.C302_main_mfun to i8*), i8* null, i8* bitcast (i32* @.C283_main_mfun to i8*), i8* bitcast (i32* @.C283_main_mfun to i8*)), !dbg !42
  call void @llvm.dbg.value(metadata i32 %3, metadata !43, metadata !DIExpression()), !dbg !33
  %4 = load i32, i32* %i1_311, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %4, metadata !35, metadata !DIExpression()), !dbg !33
  %5 = call i32 (i32, i32, ...) bitcast (i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*)(i32 %4, i32 25), !dbg !42
  call void @llvm.dbg.value(metadata i32 %5, metadata !43, metadata !DIExpression()), !dbg !33
  %6 = call i32 (...) @f90io_ldw_end(), !dbg !42
  call void @llvm.dbg.value(metadata i32 %6, metadata !43, metadata !DIExpression()), !dbg !33
  ret i32 %2, !dbg !44
}

define internal fastcc float @main_fun(i64* noalias %a) unnamed_addr !dbg !45 {
L.entry:
  call void @llvm.dbg.declare(metadata i64* %a, metadata !50, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.declare(metadata i64* undef, metadata !53, metadata !DIExpression()), !dbg !51
  %0 = bitcast i64* %a to i32*, !dbg !54
  %1 = load i32, i32* %0, align 4, !dbg !54
  %2 = sitofp i32 %1 to float, !dbg !54
  ret float %2, !dbg !59
}

declare signext i32 @f90io_ldw_end(...) local_unnamed_addr

declare signext i32 @f90io_sc_i_ldw(...) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata)

declare signext i32 @f90io_print_init(...) local_unnamed_addr

declare void @f90io_src_info03a(...) local_unnamed_addr

declare void @fort_init(...) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.module.flags = !{!14, !15}
!llvm.dbg.cu = !{!4}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "res", scope: !2, file: !3, type: !9, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "main", scope: !4, file: !3, line: 1, type: !12, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !4)
!3 = !DIFile(filename: "duplicate-dbgvalue.f90", directory: "/dir")
!4 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}
!6 = !{!0, !7, !10}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(name: "res", scope: !4, file: !3, type: !9, isLocal: true, isDefinition: true)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "res", scope: !4, file: !3, type: !9, isLocal: true, isDefinition: true)
!12 = !DISubroutineType(cc: DW_CC_program, types: !13)
!13 = !{null}
!14 = !{i32 2, !"Dwarf Version", i32 4}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !DILocation(line: 1, column: 1, scope: !17)
!17 = !DILexicalBlock(scope: !2, file: !3, line: 1, column: 1)
!18 = !DILocation(line: 3, column: 1, scope: !17)
!19 = !{!20, !20, i64 0}
!20 = !{!"t1.2", !21, i64 0}
!21 = !{!"unlimited ptr", !22, i64 0}
!22 = !{!"Flang FAA 1"}
!23 = !DILocation(line: 4, column: 1, scope: !17)
!24 = !DILocalVariable(scope: !17, file: !3, type: !9, flags: DIFlagArtificial)
!25 = !DILocation(line: 0, scope: !17)
!26 = !DILocation(line: 5, column: 1, scope: !17)
!27 = distinct !DISubprogram(name: "mfun", scope: !2, file: !3, line: 6, type: !28, scopeLine: 6, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!28 = !DISubroutineType(types: !29)
!29 = !{!30}
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, align: 64)
!31 = !DILocalVariable(arg: 1, scope: !27, file: !3, type: !32, flags: DIFlagArtificial)
!32 = !DIBasicType(name: "uinteger*8", size: 64, align: 64, encoding: DW_ATE_unsigned)
!33 = !DILocation(line: 0, scope: !34)
!34 = !DILexicalBlock(scope: !27, file: !3, line: 6, column: 1)
!35 = !DILocalVariable(name: "i1", scope: !34, file: !3, type: !9)
!36 = !DILocation(line: 8, column: 1, scope: !34)
!37 = !{!38, !38, i64 0}
!38 = !{!"t2.2", !39, i64 0}
!39 = !{!"unlimited ptr", !40, i64 0}
!40 = !{!"Flang FAA 2"}
!41 = !DILocation(line: 9, column: 1, scope: !34)
!42 = !DILocation(line: 10, column: 1, scope: !34)
!43 = !DILocalVariable(scope: !34, file: !3, type: !9, flags: DIFlagArtificial)
!44 = !DILocation(line: 11, column: 1, scope: !34)
!45 = distinct !DISubprogram(name: "fun", scope: !2, file: !3, line: 12, type: !46, scopeLine: 12, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!46 = !DISubroutineType(types: !47)
!47 = !{!48, !9}
!48 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !49, size: 64, align: 64)
!49 = !DIBasicType(name: "real", size: 32, align: 32, encoding: DW_ATE_float)
!50 = !DILocalVariable(name: "a", arg: 1, scope: !45, file: !3, type: !9)
!51 = !DILocation(line: 0, scope: !52)
!52 = !DILexicalBlock(scope: !45, file: !3, line: 12, column: 1)
!53 = !DILocalVariable(arg: 2, scope: !45, file: !3, type: !32, flags: DIFlagArtificial)
!54 = !DILocation(line: 14, column: 1, scope: !52)
!55 = !{!56, !56, i64 0}
!56 = !{!"t3.2", !57, i64 0}
!57 = !{!"unlimited ptr", !58, i64 0}
!58 = !{!"Flang FAA 3"}
!59 = !DILocation(line: 15, column: 1, scope: !52)
