; RUN: opt < %s -mem2reg -S | FileCheck %s

define double @testfunc(i32 %i, double %j) nounwind ssp {
entry:
  %i_addr = alloca i32                            ; <i32*> [#uses=2]
  %j_addr = alloca double                         ; <double*> [#uses=2]
  %retval = alloca double                         ; <double*> [#uses=2]
  %0 = alloca double                              ; <double*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata !{i32* %i_addr}, metadata !0), !dbg !8
; CHECK: call void @llvm.dbg.value(metadata !{i32 %i}, i64 0, metadata !0)
; CHECK: call void @llvm.dbg.value(metadata !{double %j}, i64 0, metadata !9)
  store i32 %i, i32* %i_addr
  call void @llvm.dbg.declare(metadata !{double* %j_addr}, metadata !9), !dbg !8
  store double %j, double* %j_addr
  %1 = load i32* %i_addr, align 4, !dbg !10       ; <i32> [#uses=1]
  %2 = add nsw i32 %1, 1, !dbg !10                ; <i32> [#uses=1]
  %3 = sitofp i32 %2 to double, !dbg !10          ; <double> [#uses=1]
  %4 = load double* %j_addr, align 8, !dbg !10    ; <double> [#uses=1]
  %5 = fadd double %3, %4, !dbg !10               ; <double> [#uses=1]
  store double %5, double* %0, align 8, !dbg !10
  %6 = load double* %0, align 8, !dbg !10         ; <double> [#uses=1]
  store double %6, double* %retval, align 8, !dbg !10
  br label %return, !dbg !10

return:                                           ; preds = %entry
  %retval1 = load double* %retval, !dbg !10       ; <double> [#uses=1]
  ret double %retval1, !dbg !10
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

!0 = metadata !{i32 524545, metadata !1, metadata !"i", metadata !2, i32 2, metadata !7} ; [ DW_TAG_arg_variable ]
!1 = metadata !{i32 524334, i32 0, metadata !2, metadata !"testfunc", metadata !"testfunc", metadata !"testfunc", metadata !2, i32 2, metadata !4, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 524329, metadata !"testfunc.c", metadata !"/tmp", metadata !3} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 524305, i32 0, i32 1, metadata !"testfunc.c", metadata !"/tmp", metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!4 = metadata !{i32 524309, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !5, i32 0, null} ; [ DW_TAG_subroutine_type ]
!5 = metadata !{metadata !6, metadata !7, metadata !6}
!6 = metadata !{i32 524324, metadata !2, metadata !"double", metadata !2, i32 0, i64 64, i64 64, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ]
!7 = metadata !{i32 524324, metadata !2, metadata !"int", metadata !2, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!8 = metadata !{i32 2, i32 0, metadata !1, null}
!9 = metadata !{i32 524545, metadata !1, metadata !"j", metadata !2, i32 2, metadata !6} ; [ DW_TAG_arg_variable ]
!10 = metadata !{i32 3, i32 0, metadata !11, null}
!11 = metadata !{i32 524299, metadata !1, i32 2, i32 0} ; [ DW_TAG_lexical_block ]

