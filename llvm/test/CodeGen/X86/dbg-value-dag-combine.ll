; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"
; PR 9817


declare  <4 x i32> @__amdil_get_global_id_int()
declare  void @llvm.dbg.value(metadata , i64 , metadata )
define void @__OpenCL_test_kernel(i32 addrspace(1)* %ip) nounwind {
entry:
  call void @llvm.dbg.value(metadata !{i32 addrspace(1)* %ip}, i64 0, metadata
!7), !dbg !8
  %0 = call <4 x i32> @__amdil_get_global_id_int() nounwind
  %1 = extractelement <4 x i32> %0, i32 0
  call void @llvm.dbg.value(metadata !{i32 %1}, i64 0, metadata !9), !dbg !11
  call void @llvm.dbg.value(metadata !12, i64 0, metadata !13), !dbg !14
  %tmp2 = load i32 addrspace(1)* %ip, align 4, !dbg !15
  %tmp3 = add i32 0, %tmp2, !dbg !15
; CHECK:  ##DEBUG_VALUE: idx <- EAX+0
  call void @llvm.dbg.value(metadata !{i32 %tmp3}, i64 0, metadata !13), !dbg
!15
  %arrayidx = getelementptr i32 addrspace(1)* %ip, i32 %1, !dbg !16
  store i32 %tmp3, i32 addrspace(1)* %arrayidx, align 4, !dbg !16
  ret void, !dbg !17
}
!llvm.dbg.sp = !{!0}

!0 = metadata !{i32 589870, i32 0, metadata !1, metadata
!"__OpenCL_test_kernel", metadata !"__OpenCL_test_kernel", metadata
!"__OpenCL_test_kernel", metadata !1, i32 2, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 0, i1 false, null} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 589865, metadata !"OCL6368.tmp.cl", metadata !"E:\5CUsers\5Cmvillmow.AMD\5CAppData\5CLocal\5CTemp", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, i32 0, i32 1, metadata !"OCL6368.tmp.cl", metadata !"E:\5CUsers\5Cmvillmow.AMD\5CAppData\5CLocal\5CTemp", metadata !"clc", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{null, metadata !5}
!5 = metadata !{i32 589839, metadata !2, metadata !"", null, i32 0, i64 32, i64 32, i64 0, i32 0, metadata !6} ; [ DW_TAG_pointer_type ]
!6 = metadata !{i32 589860, metadata !2, metadata !"unsigned int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!7 = metadata !{i32 590081, metadata !0, metadata !"ip", metadata !1, i32 1, metadata !5, i32 0} ; [ DW_TAG_arg_variable ]
!8 = metadata !{i32 1, i32 42, metadata !0, null}
!9 = metadata !{i32 590080, metadata !10, metadata !"gid", metadata !1, i32 3, metadata !6, i32 0} ; [ DW_TAG_auto_variable ]
!10 = metadata !{i32 589835, metadata !0, i32 2, i32 1, metadata !1, i32 0} ; [ DW_TAG_lexical_block ]
!11 = metadata !{i32 3, i32 41, metadata !10, null}
!12 = metadata !{i32 0}
!13 = metadata !{i32 590080, metadata !10, metadata !"idx", metadata !1, i32 4, metadata !6, i32 0} ; [ DW_TAG_auto_variable ]
!14 = metadata !{i32 4, i32 20, metadata !10, null}
!15 = metadata !{i32 5, i32 15, metadata !10, null}
!16 = metadata !{i32 6, i32 18, metadata !10, null}
!17 = metadata !{i32 7, i32 1, metadata !0, null}

