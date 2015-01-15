; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"
; PR 9879

; CHECK: ##DEBUG_VALUE: tid <-
%0 = type { i8*, i8*, i8*, i8*, i32 }

@sgv = internal addrspace(2) constant [1 x i8] zeroinitializer
@fgv = internal addrspace(2) constant [1 x i8] zeroinitializer
@lvgv = internal constant [0 x i8*] zeroinitializer
@llvm.global.annotations = appending global [1 x %0] [%0 { i8* bitcast (void (i32 addrspace(1)*)* @__OpenCL_nbt02_kernel to i8*), i8* addrspacecast ([1 x i8] addrspace(2)* @sgv to i8*), i8* addrspacecast ([1 x i8] addrspace(2)* @fgv to i8*), i8* bitcast ([0 x i8*]* @lvgv to i8*), i32 0 }], section "llvm.metadata"

define void @__OpenCL_nbt02_kernel(i32 addrspace(1)* %ip) nounwind {
entry:
  call void @llvm.dbg.value(metadata i32 addrspace(1)* %ip, i64 0, metadata !8, metadata !{!"0x102"}), !dbg !9
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
  call void @llvm.dbg.value(metadata i32 %6, i64 0, metadata !10, metadata !{!"0x102"}), !dbg !12
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
  call void @llvm.dbg.value(metadata i32 %13, i64 0, metadata !13, metadata !{!"0x102"}), !dbg !14
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
  call void @llvm.dbg.value(metadata i32 %20, i64 0, metadata !15, metadata !{!"0x102"}), !dbg !16
  %tmp5 = add i32 %6, %13, !dbg !17
  %tmp7 = add i32 %tmp5, %20, !dbg !17
  store i32 %tmp7, i32 addrspace(1)* %ip, align 4, !dbg !17
  br label %return, !dbg !17

return:                                           ; preds = %get_local_size.exit
  ret void, !dbg !18
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare <4 x i32> @__amdil_get_local_size_int() nounwind

declare <4 x i32> @__amdil_get_local_id_int() nounwind

declare <4 x i32> @__amdil_get_global_id_int() nounwind

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!22}

!0 = !{!"0x2e\00__OpenCL_nbt02_kernel\00__OpenCL_nbt02_kernel\00__OpenCL_nbt02_kernel\002\000\001\000\006\000\000\000", !20, !1, !3, null, void (i32 addrspace(1)*)* @__OpenCL_nbt02_kernel, null, null, null} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 0] [__OpenCL_nbt02_kernel]
!1 = !{!"0x29", !20} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\001\00clc\000\00\000\00\001", !20, !21, !21, !19, null,  null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !20, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{null, !5}
!5 = !{!"0xf\00\000\0032\0032\000\000", null, !2, !6} ; [ DW_TAG_pointer_type ]
!6 = !{!"0x16\00uint\000\000\000\000\000", !20, !2, !7} ; [ DW_TAG_typedef ]
!7 = !{!"0x24\00unsigned int\000\0032\0032\000\000\007", null, !2} ; [ DW_TAG_base_type ]
!8 = !{!"0x101\00ip\001\000", !0, !1, !5} ; [ DW_TAG_arg_variable ]
!9 = !MDLocation(line: 1, column: 32, scope: !0)
!10 = !{!"0x100\00tid\003\000", !11, !1, !6} ; [ DW_TAG_auto_variable ]
!11 = !{!"0xb\002\001\001", !1, !0} ; [ DW_TAG_lexical_block ]
!12 = !MDLocation(line: 5, column: 24, scope: !11)
!13 = !{!"0x100\00gid\003\000", !11, !1, !6} ; [ DW_TAG_auto_variable ]
!14 = !MDLocation(line: 6, column: 25, scope: !11)
!15 = !{!"0x100\00lsz\003\000", !11, !1, !6} ; [ DW_TAG_auto_variable ]
!16 = !MDLocation(line: 7, column: 26, scope: !11)
!17 = !MDLocation(line: 9, column: 24, scope: !11)
!18 = !MDLocation(line: 10, column: 1, scope: !0)
!19 = !{!0}
!20 = !{!"OCLlLwTXZ.cl", !"/tmp"}
!21 = !{i32 0}
!22 = !{i32 1, !"Debug Info Version", i32 2}
