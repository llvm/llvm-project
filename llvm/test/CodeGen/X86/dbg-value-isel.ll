; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"
; PR 9879

; CHECK: ##DEBUG_VALUE: tid <-
%0 = type { i8*, i8*, i8*, i8*, i32 }

@sgv = internal addrspace(2) constant [1 x i8] zeroinitializer
@fgv = internal addrspace(2) constant [1 x i8] zeroinitializer
@lvgv = internal constant [0 x i8*] zeroinitializer
@llvm.global.annotations = appending global [1 x %0] [%0 { i8* bitcast (void (i32 addrspace(1)*)* @__OpenCL_nbt02_kernel to i8*), i8* bitcast ([1 x i8] addrspace(2)* @sgv to i8*), i8* bitcast ([1 x i8] addrspace(2)* @fgv to i8*), i8* bitcast ([0 x i8*]* @lvgv to i8*), i32 0 }], section "llvm.metadata"

define void @__OpenCL_nbt02_kernel(i32 addrspace(1)* %ip) nounwind {
entry:
  call void @llvm.dbg.value(metadata !{i32 addrspace(1)* %ip}, i64 0, metadata !8), !dbg !9
  %0 = call <4 x i32> @__amdil_get_local_id_int() nounwind
  %1 = extractelement <4 x i32> %0, i32 0
  br label %2

; <label>:2                                       ; preds = %entry
  %3 = phi i32 [ %1, %entry ]
  br label %4

; <label>:4                                       ; preds = %2
  %5 = phi i32 [ %3, %2 ]
  br label %get_local_id.exit

get_local_id.exit:                                ; preds = %4
  %6 = phi i32 [ %5, %4 ]
  call void @llvm.dbg.value(metadata !{i32 %6}, i64 0, metadata !10), !dbg !12
  %7 = call <4 x i32> @__amdil_get_global_id_int() nounwind, !dbg !12
  %8 = extractelement <4 x i32> %7, i32 0, !dbg !12
  br label %9

; <label>:9                                       ; preds = %get_local_id.exit
  %10 = phi i32 [ %8, %get_local_id.exit ]
  br label %11

; <label>:11                                      ; preds = %9
  %12 = phi i32 [ %10, %9 ]
  br label %get_global_id.exit

get_global_id.exit:                               ; preds = %11
  %13 = phi i32 [ %12, %11 ]
  call void @llvm.dbg.value(metadata !{i32 %13}, i64 0, metadata !13), !dbg !14
  %14 = call <4 x i32> @__amdil_get_local_size_int() nounwind
  %15 = extractelement <4 x i32> %14, i32 0
  br label %16

; <label>:16                                      ; preds = %get_global_id.exit
  %17 = phi i32 [ %15, %get_global_id.exit ]
  br label %18

; <label>:18                                      ; preds = %16
  %19 = phi i32 [ %17, %16 ]
  br label %get_local_size.exit

get_local_size.exit:                              ; preds = %18
  %20 = phi i32 [ %19, %18 ]
  call void @llvm.dbg.value(metadata !{i32 %20}, i64 0, metadata !15), !dbg !16
  %tmp5 = add i32 %6, %13, !dbg !17
  %tmp7 = add i32 %tmp5, %20, !dbg !17
  store i32 %tmp7, i32 addrspace(1)* %ip, align 4, !dbg !17
  br label %return, !dbg !17

return:                                           ; preds = %get_local_size.exit
  ret void, !dbg !18
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare <4 x i32> @__amdil_get_local_size_int() nounwind

declare <4 x i32> @__amdil_get_local_id_int() nounwind

declare <4 x i32> @__amdil_get_global_id_int() nounwind

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.sp = !{!0}

!0 = metadata !{i32 589870, i32 0, metadata !1, metadata !"__OpenCL_nbt02_kernel", metadata !"__OpenCL_nbt02_kernel", metadata !"__OpenCL_nbt02_kernel", metadata !1, i32 2, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 0, i1 false, null} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 589865, metadata !"OCLlLwTXZ.cl", metadata !"/tmp", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, i32 0, i32 1, metadata !"OCLlLwTXZ.cl", metadata !"/tmp", metadata !"clc", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{null, metadata !5}
!5 = metadata !{i32 589839, metadata !2, metadata !"", null, i32 0, i64 32, i64 32, i64 0, i32 0, metadata !6} ; [ DW_TAG_pointer_type ]
!6 = metadata !{i32 589846, metadata !2, metadata !"uint", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !7} ; [ DW_TAG_typedef ]
!7 = metadata !{i32 589860, metadata !2, metadata !"unsigned int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!8 = metadata !{i32 590081, metadata !0, metadata !"ip", metadata !1, i32 1, metadata !5, i32 0} ; [ DW_TAG_arg_variable ]
!9 = metadata !{i32 1, i32 32, metadata !0, null}
!10 = metadata !{i32 590080, metadata !11, metadata !"tid", metadata !1, i32 3, metadata !6, i32 0} ; [ DW_TAG_auto_variable ]
!11 = metadata !{i32 589835, metadata !0, i32 2, i32 1, metadata !1, i32 1} ; [ DW_TAG_lexical_block ]
!12 = metadata !{i32 5, i32 24, metadata !11, null}
!13 = metadata !{i32 590080, metadata !11, metadata !"gid", metadata !1, i32 3, metadata !6, i32 0} ; [ DW_TAG_auto_variable ]
!14 = metadata !{i32 6, i32 25, metadata !11, null}
!15 = metadata !{i32 590080, metadata !11, metadata !"lsz", metadata !1, i32 3, metadata !6, i32 0} ; [ DW_TAG_auto_variable ]
!16 = metadata !{i32 7, i32 26, metadata !11, null}
!17 = metadata !{i32 9, i32 24, metadata !11, null}
!18 = metadata !{i32 10, i32 1, metadata !0, null}

