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
; CHECK:  ##DEBUG_VALUE: idx <- E{{..$}}
  call void @llvm.dbg.value(metadata !{i32 %tmp3}, i64 0, metadata !13), !dbg
!15
  %arrayidx = getelementptr i32 addrspace(1)* %ip, i32 %1, !dbg !16
  store i32 %tmp3, i32 addrspace(1)* %arrayidx, align 4, !dbg !16
  ret void, !dbg !17
}
!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!20}

!0 = metadata !{i32 786478, metadata !19, metadata !1, metadata !"__OpenCL_test_kernel", metadata !"__OpenCL_test_kernel", metadata !"__OpenCL_test_kernel", i32 2, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, null, null, null, null, i32 0} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 0] [__OpenCL_test_kernel]
!1 = metadata !{i32 786473, metadata !19} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 786449, metadata !19, i32 1, metadata !"clc", i1 false, metadata !"", i32 0, metadata !12, metadata !12, metadata !18, null,  null, null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 786453, metadata !19, metadata !1, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{null, metadata !5}
!5 = metadata !{i32 786447, null, metadata !2, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !6} ; [ DW_TAG_pointer_type ]
!6 = metadata !{i32 786468, null, metadata !2, metadata !"unsigned int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!7 = metadata !{i32 786689, metadata !0, metadata !"ip", metadata !1, i32 1, metadata !5, i32 0, null} ; [ DW_TAG_arg_variable ]
!8 = metadata !{i32 1, i32 42, metadata !0, null}
!9 = metadata !{i32 786688, metadata !10, metadata !"gid", metadata !1, i32 3, metadata !6, i32 0, null} ; [ DW_TAG_auto_variable ]
!10 = metadata !{i32 786443, metadata !19, metadata !0, i32 2, i32 1, i32 0} ; [ DW_TAG_lexical_block ]
!11 = metadata !{i32 3, i32 41, metadata !10, null}
!12 = metadata !{i32 0}
!13 = metadata !{i32 786688, metadata !10, metadata !"idx", metadata !1, i32 4, metadata !6, i32 0, null} ; [ DW_TAG_auto_variable ]
!14 = metadata !{i32 4, i32 20, metadata !10, null}
!15 = metadata !{i32 5, i32 15, metadata !10, null}
!16 = metadata !{i32 6, i32 18, metadata !10, null}
!17 = metadata !{i32 7, i32 1, metadata !0, null}
!18 = metadata !{metadata !0}
!19 = metadata !{metadata !"OCL6368.tmp.cl", metadata !"E:\5CUsers\5Cmvillmow.AMD\5CAppData\5CLocal\5CTemp"}
!20 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
