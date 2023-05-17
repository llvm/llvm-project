; RUN: opt -S -p globalopt < %s | FileCheck %s

;; Generated at -g from:
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;MODULE mymod
;;   IMPLICIT NONE
;;   INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(14, 200)
;;CONTAINS
;;   FUNCTION foo(arg)
;;      REAL(KIND=dp), DIMENSION(3, 3), INTENT(IN)         :: arg
;;      INTEGER, DIMENSION(3)                              :: foo
;;      foo = bar(arg)
;;   END FUNCTION foo
;;   FUNCTION bar(arg)
;;      REAL(KIND=dp), DIMENSION(3, 3), INTENT(IN)         :: arg
;;      INTEGER, DIMENSION(3)                              :: bar
;;      REAL(KIND=dp)                                      :: rvar(3)
;;      rvar = arg(1,:)
;;      IF(rvar(1) == 0._dp) print *,"IF"
;;      bar = FLOOR(rvar)
;;   END FUNCTION bar
;;END MODULE mymod
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

%struct.BSS3 = type <{ [24 x i8] }>

@.BSS3 = internal unnamed_addr global %struct.BSS3 zeroinitializer, align 32, !dbg !0, !dbg !7, !dbg !29
;CHECK: @.BSS3.0 = internal unnamed_addr global double 0.000000e+00, align 32, !dbg ![[GVE1:.*]], !dbg ![[GVE2:.*]]
;CHECK: @.BSS3.1 = internal unnamed_addr global double 0.000000e+00, align 32, !dbg ![[GVE3:.*]], !dbg ![[GVE4:.*]], !dbg ![[GVE6:.*]]
;CHECK: @.BSS3.2 = internal unnamed_addr global double 0.000000e+00, align 16, !dbg ![[GVE5:.*]]

@.C363_mymod_bar_ = internal constant [2 x i8] c"IF"
@.C330_mymod_bar_ = internal constant i32 0
@.C360_mymod_bar_ = internal constant i32 6
@.C357_mymod_bar_ = internal constant [9 x i8] c"frag1.f90"
@.C359_mymod_bar_ = internal constant i32 15

define float @mymod_() local_unnamed_addr {
.L.entry:
  ret float undef
}

define void @mymod_foo_(i64* noalias nocapture writeonly %foo, i64* noalias nocapture readonly %arg) local_unnamed_addr !dbg !22 {
L.entry:
  tail call void @llvm.experimental.noalias.scope.decl(metadata !23)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !26)
  %0 = bitcast i64* %arg to double*
  %1 = load double, double* %0, align 8, !noalias !23
  store double %1, double* bitcast (%struct.BSS3* @.BSS3 to double*), align 32, !noalias !28
  %2 = getelementptr i64, i64* %arg, i64 3
  %3 = bitcast i64* %2 to double*
  %4 = load double, double* %3, align 8, !alias.scope !26, !noalias !23
  store double %4, double* bitcast (i8* getelementptr inbounds (%struct.BSS3, %struct.BSS3* @.BSS3, i64 0, i32 0, i64 8) to double*), align 8, !noalias !28
  %5 = getelementptr i64, i64* %arg, i64 6
  %6 = bitcast i64* %5 to double*
  %7 = load double, double* %6, align 8, !alias.scope !26, !noalias !23
  store double %7, double* bitcast (i8* getelementptr inbounds (%struct.BSS3, %struct.BSS3* @.BSS3, i64 0, i32 0, i64 16) to double*), align 16, !noalias !28
  %8 = fcmp une double %1, 0.000000e+00
  br i1 %8, label %L.LB3_377.i, label %L.LB3_417.i

L.LB3_417.i:                                      ; preds = %L.entry
  tail call void (i8*, i8*, i64, ...) bitcast (void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*)(i8* bitcast (i32* @.C359_mymod_bar_ to i8*), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.C357_mymod_bar_, i64 0, i64 0), i64 9), !noalias !28
  %9 = tail call i32 (i8*, i8*, i8*, i8*, ...) bitcast (i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*)(i8* bitcast (i32* @.C360_mymod_bar_ to i8*), i8* null, i8* bitcast (i32* @.C330_mymod_bar_ to i8*), i8* bitcast (i32* @.C330_mymod_bar_ to i8*)), !noalias !28
  %10 = tail call i32 (i8*, i32, i64, ...) bitcast (i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*)(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.C363_mymod_bar_, i64 0, i64 0), i32 14, i64 2), !noalias !28
  %11 = tail call i32 (...) @f90io_ldw_end(), !noalias !28
  %.pre = load double, double* bitcast (%struct.BSS3* @.BSS3 to double*), align 32, !noalias !28
  %.pre1 = load double, double* bitcast (i8* getelementptr inbounds (%struct.BSS3, %struct.BSS3* @.BSS3, i64 0, i32 0, i64 8) to double*), align 8, !noalias !28
  %.pre2 = load double, double* bitcast (i8* getelementptr inbounds (%struct.BSS3, %struct.BSS3* @.BSS3, i64 0, i32 0, i64 16) to double*), align 16, !noalias !28
  br label %L.LB3_377.i

L.LB3_377.i:                                      ; preds = %L.LB3_417.i, %L.entry
  %12 = phi double [ %.pre2, %L.LB3_417.i ], [ %7, %L.entry ]
  %13 = phi double [ %.pre1, %L.LB3_417.i ], [ %4, %L.entry ]
  %14 = phi double [ %.pre, %L.LB3_417.i ], [ %1, %L.entry ]
  %15 = bitcast i64* %foo to i8*
  %16 = tail call double @llvm.floor.f64(double %14)
  %17 = fptosi double %16 to i32
  %18 = bitcast i64* %foo to i32*
  store i32 %17, i32* %18, align 4, !alias.scope !23, !noalias !26
  %19 = tail call double @llvm.floor.f64(double %13)
  %20 = fptosi double %19 to i32
  %21 = getelementptr i8, i8* %15, i64 4
  %22 = bitcast i8* %21 to i32*
  store i32 %20, i32* %22, align 4, !alias.scope !23, !noalias !26
  %23 = tail call double @llvm.floor.f64(double %12)
  %24 = fptosi double %23 to i32
  %25 = getelementptr i64, i64* %foo, i64 1
  %26 = bitcast i64* %25 to i32*
  store i32 %24, i32* %26, align 4, !alias.scope !23, !noalias !26
  ret void
}

define void @mymod_bar_(i64* noalias nocapture writeonly %bar, i64* noalias nocapture readonly %arg) local_unnamed_addr !dbg !9 {
L.entry:
  %0 = bitcast i64* %arg to double*
  %1 = load double, double* %0, align 8
  store double %1, double* bitcast (%struct.BSS3* @.BSS3 to double*), align 32
  %2 = getelementptr i64, i64* %arg, i64 3
  %3 = bitcast i64* %2 to double*
  %4 = load double, double* %3, align 8
  store double %4, double* bitcast (i8* getelementptr inbounds (%struct.BSS3, %struct.BSS3* @.BSS3, i64 0, i32 0, i64 8) to double*), align 8
  %5 = getelementptr i64, i64* %arg, i64 6
  %6 = bitcast i64* %5 to double*
  %7 = load double, double* %6, align 8
  store double %7, double* bitcast (i8* getelementptr inbounds (%struct.BSS3, %struct.BSS3* @.BSS3, i64 0, i32 0, i64 16) to double*), align 16
  %8 = fcmp une double %1, 0.000000e+00
  br i1 %8, label %L.LB3_377, label %L.LB3_417

L.LB3_417:                                        ; preds = %L.entry
  tail call void (i8*, i8*, i64, ...) bitcast (void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*)(i8* bitcast (i32* @.C359_mymod_bar_ to i8*), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.C357_mymod_bar_, i64 0, i64 0), i64 9)
  %9 = tail call i32 (i8*, i8*, i8*, i8*, ...) bitcast (i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*)(i8* bitcast (i32* @.C360_mymod_bar_ to i8*), i8* null, i8* bitcast (i32* @.C330_mymod_bar_ to i8*), i8* bitcast (i32* @.C330_mymod_bar_ to i8*))
  %10 = tail call i32 (i8*, i32, i64, ...) bitcast (i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*)(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.C363_mymod_bar_, i64 0, i64 0), i32 14, i64 2)
  %11 = tail call i32 (...) @f90io_ldw_end()
  %.pre = load double, double* bitcast (%struct.BSS3* @.BSS3 to double*), align 32
  %.pre1 = load double, double* bitcast (i8* getelementptr inbounds (%struct.BSS3, %struct.BSS3* @.BSS3, i64 0, i32 0, i64 8) to double*), align 8
  %.pre2 = load double, double* bitcast (i8* getelementptr inbounds (%struct.BSS3, %struct.BSS3* @.BSS3, i64 0, i32 0, i64 16) to double*), align 16
  br label %L.LB3_377

L.LB3_377:                                        ; preds = %L.LB3_417, %L.entry
  %12 = phi double [ %.pre2, %L.LB3_417 ], [ %7, %L.entry ]
  %13 = phi double [ %.pre1, %L.LB3_417 ], [ %4, %L.entry ]
  %14 = phi double [ %.pre, %L.LB3_417 ], [ %1, %L.entry ]
  %15 = bitcast i64* %bar to i8*
  %16 = tail call double @llvm.floor.f64(double %14)
  %17 = fptosi double %16 to i32
  %18 = bitcast i64* %bar to i32*
  store i32 %17, i32* %18, align 4
  %19 = tail call double @llvm.floor.f64(double %13)
  %20 = fptosi double %19 to i32
  %21 = getelementptr i8, i8* %15, i64 4
  %22 = bitcast i8* %21 to i32*
  store i32 %20, i32* %22, align 4
  %23 = tail call double @llvm.floor.f64(double %12)
  %24 = fptosi double %23 to i32
  %25 = getelementptr i64, i64* %bar, i64 1
  %26 = bitcast i64* %25 to i32*
  store i32 %24, i32* %26, align 4
  ret void
}

declare signext i32 @f90io_ldw_end(...) local_unnamed_addr

declare signext i32 @f90io_sc_ch_ldw(...) local_unnamed_addr

declare signext i32 @f90io_print_init(...) local_unnamed_addr

declare void @f90io_src_info03a(...) local_unnamed_addr

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.floor.f64(double)

; Function Attrs: inaccessiblememonly nocallback nofree nosync nounwind willreturn
declare void @llvm.experimental.noalias.scope.decl(metadata)

!llvm.module.flags = !{!20, !21}
!llvm.dbg.cu = !{!3}

; CHECK-DAG: ![[GVE1]] = !DIGlobalVariableExpression(var: ![[GV1:.*]], expr: !DIExpression(DW_OP_LLVM_fragment, 0, 64))
; CHECK-DAG: ![[GV1]] = distinct !DIGlobalVariable(name: "bar1"
; CHECK-DAG: ![[GVE2]] = !DIGlobalVariableExpression(var: ![[GV2:.*]], expr: !DIExpression(DW_OP_LLVM_fragment, 0, 64))
; CHECK-DAG: ![[GV2]] = distinct !DIGlobalVariable(name: "rvar"
; CHECK-DAG: ![[GVE3]] = !DIGlobalVariableExpression(var: ![[GV1]], expr: !DIExpression(DW_OP_LLVM_fragment, 64, 32))
; CHECK-DAG: ![[GVE4]] = !DIGlobalVariableExpression(var: ![[GV2]], expr: !DIExpression(DW_OP_LLVM_fragment, 64, 64))
; CHECK-DAG: ![[GVE5]] = !DIGlobalVariableExpression(var: ![[GV2]], expr: !DIExpression(DW_OP_LLVM_fragment, 128, 64))
; CHECK-DAG: ![[GVE6]] = !DIGlobalVariableExpression(var: ![[GV3:.*]], expr: !DIExpression(DW_OP_LLVM_fragment, 32, 32))
; CHECK-DAG: ![[GV3]] = distinct !DIGlobalVariable(name: "ivar"

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "bar1", scope: !2, file: !4, type: !12, isLocal: true, isDefinition: true)
!2 = !DIModule(scope: !3, name: "mymod", isDecl: true)
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: true, flags: "'+flang -g -O3 -S -emit-llvm -o -O1'", runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5, nameTableKind: None)
!4 = !DIFile(filename: "global-sra-struct-part-overlap-segment_0.f90", directory: "/tmp")
!5 = !{}
!6 = !{!0, !7, !29}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(name: "rvar", scope: !9, file: !4, line: 13, type: !19, isLocal: true, isDefinition: true)
!9 = distinct !DISubprogram(name: "bar", scope: !2, file: !4, line: 10, type: !10, scopeLine: 10, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !3)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12, !16}
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 96, align: 32, elements: !14)
!13 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = !{!15}
!15 = !DISubrange(lowerBound: 1, upperBound: 3)
!16 = !DICompositeType(tag: DW_TAG_array_type, baseType: !17, size: 576, align: 64, elements: !18)
!17 = !DIBasicType(name: "double precision", size: 64, align: 64, encoding: DW_ATE_float)
!18 = !{!15, !15}
!19 = !DICompositeType(tag: DW_TAG_array_type, baseType: !17, size: 192, align: 64, elements: !14)
!20 = !{i32 2, !"Dwarf Version", i32 4}
!21 = !{i32 2, !"Debug Info Version", i32 3}
!22 = distinct !DISubprogram(name: "foo", scope: !2, file: !4, line: 5, type: !10, scopeLine: 5, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !3)
!23 = !{!24}
!24 = distinct !{!24, !25, !"mymod_bar_: %bar"}
!25 = distinct !{!25, !"mymod_bar_"}
!26 = !{!27}
!27 = distinct !{!27, !25, !"mymod_bar_: %arg"}
!28 = !{!24, !27}
!29 = !DIGlobalVariableExpression(var: !30, expr: !DIExpression(DW_OP_plus_uconst, 4))
!30 = distinct !DIGlobalVariable(name: "ivar", scope: !9, file: !4, line: 13, type: !31, isLocal: true, isDefinition: true)
!31 = !DIBasicType(name: "integer*8", size: 64, align: 32, encoding: DW_ATE_signed)
