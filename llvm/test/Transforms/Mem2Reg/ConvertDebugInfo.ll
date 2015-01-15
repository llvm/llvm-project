; RUN: opt < %s -mem2reg -S | FileCheck %s

define double @testfunc(i32 %i, double %j) nounwind ssp {
entry:
  %i_addr = alloca i32                            ; <i32*> [#uses=2]
  %j_addr = alloca double                         ; <double*> [#uses=2]
  %retval = alloca double                         ; <double*> [#uses=2]
  %0 = alloca double                              ; <double*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata i32* %i_addr, metadata !0, metadata !{}), !dbg !8
; CHECK: call void @llvm.dbg.value(metadata i32 %i, i64 0, metadata ![[IVAR:[0-9]*]], metadata {{.*}})
; CHECK: call void @llvm.dbg.value(metadata double %j, i64 0, metadata ![[JVAR:[0-9]*]], metadata {{.*}})
; CHECK: ![[IVAR]] = {{.*}} ; [ DW_TAG_arg_variable ] [i]
; CHECK: ![[JVAR]] = {{.*}} ; [ DW_TAG_arg_variable ] [j]
  store i32 %i, i32* %i_addr
  call void @llvm.dbg.declare(metadata double* %j_addr, metadata !9, metadata !{}), !dbg !8
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

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!14}

!0 = !{!"0x101\00i\002\000", !1, !2, !7} ; [ DW_TAG_arg_variable ]
!1 = !{!"0x2e\00testfunc\00testfunc\00testfunc\002\000\001\000\006\000\000\002", !12, !2, !4, null, double (i32, double)* @testfunc, null, null, null} ; [ DW_TAG_subprogram ]
!2 = !{!"0x29", !12} ; [ DW_TAG_file_type ]
!3 = !{!"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\001\00\000\00\000", !12, !13, !13, null, null, null} ; [ DW_TAG_compile_unit ]
!4 = !{!"0x15\00\000\000\000\000\000\000", !12, !2, null, !5, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = !{!6, !7, !6}
!6 = !{!"0x24\00double\000\0064\0064\000\000\004", !12, !2} ; [ DW_TAG_base_type ]
!7 = !{!"0x24\00int\000\0032\0032\000\000\005", !12, !2} ; [ DW_TAG_base_type ]
!8 = !MDLocation(line: 2, scope: !1)
!9 = !{!"0x101\00j\002\000", !1, !2, !6} ; [ DW_TAG_arg_variable ]
!10 = !MDLocation(line: 3, scope: !11)
!11 = !{!"0xb\002\000\000", !12, !1} ; [ DW_TAG_lexical_block ]
!12 = !{!"testfunc.c", !"/tmp"}
!13 = !{i32 0}
!14 = !{i32 1, !"Debug Info Version", i32 2}
