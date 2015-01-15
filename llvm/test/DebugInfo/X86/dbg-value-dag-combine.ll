; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"
; PR 9817


declare <4 x i32> @__amdil_get_global_id_int()
declare void @llvm.dbg.value(metadata, i64, metadata, metadata)
define void @__OpenCL_test_kernel(i32 addrspace(1)* %ip) nounwind {
entry:
  call void @llvm.dbg.value(metadata i32 addrspace(1)* %ip, i64 0, metadata !7, metadata !{!"0x102"}), !dbg !8
  %0 = call <4 x i32> @__amdil_get_global_id_int() nounwind
  %1 = extractelement <4 x i32> %0, i32 0
  call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !9, metadata !{!"0x102"}), !dbg !11
  call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !13, metadata !{!"0x102"}), !dbg !14
  %tmp2 = load i32 addrspace(1)* %ip, align 4, !dbg !15
  %tmp3 = add i32 0, %tmp2, !dbg !15
; CHECK:  ##DEBUG_VALUE: idx <- E{{..$}}
  call void @llvm.dbg.value(metadata i32 %tmp3, i64 0, metadata !13, metadata !{!"0x102"}), !dbg !15
  %arrayidx = getelementptr i32 addrspace(1)* %ip, i32 %1, !dbg !16
  store i32 %tmp3, i32 addrspace(1)* %arrayidx, align 4, !dbg !16
  ret void, !dbg !17
}
!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!20}

!0 = !{!"0x2e\00__OpenCL_test_kernel\00__OpenCL_test_kernel\00__OpenCL_test_kernel\002\000\001\000\006\000\000\000", !19, !1, !3, null, void (i32 addrspace(1)*)* @__OpenCL_test_kernel, null, null, null} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 0] [__OpenCL_test_kernel]
!1 = !{!"0x29", !19} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\001\00clc\000\00\000\00\001", !19, !12, !12, !18, null,  null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !19, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{null, !5}
!5 = !{!"0xf\00\000\0032\0032\000\000", null, !2, !6} ; [ DW_TAG_pointer_type ]
!6 = !{!"0x24\00unsigned int\000\0032\0032\000\000\007", null, !2} ; [ DW_TAG_base_type ]
!7 = !{!"0x101\00ip\001\000", !0, !1, !5} ; [ DW_TAG_arg_variable ]
!8 = !MDLocation(line: 1, column: 42, scope: !0)
!9 = !{!"0x100\00gid\003\000", !10, !1, !6} ; [ DW_TAG_auto_variable ]
!10 = !{!"0xb\002\001\000", !19, !0} ; [ DW_TAG_lexical_block ]
!11 = !MDLocation(line: 3, column: 41, scope: !10)
!12 = !{i32 0}
!13 = !{!"0x100\00idx\004\000", !10, !1, !6} ; [ DW_TAG_auto_variable ]
!14 = !MDLocation(line: 4, column: 20, scope: !10)
!15 = !MDLocation(line: 5, column: 15, scope: !10)
!16 = !MDLocation(line: 6, column: 18, scope: !10)
!17 = !MDLocation(line: 7, column: 1, scope: !0)
!18 = !{!0}
!19 = !{!"OCL6368.tmp.cl", !"E:\5CUsers\5Cmvillmow.AMD\5CAppData\5CLocal\5CTemp"}
!20 = !{i32 1, !"Debug Info Version", i32 2}
