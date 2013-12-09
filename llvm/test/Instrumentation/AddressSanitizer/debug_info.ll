; RUN: opt < %s -asan -asan-module -S | FileCheck %s

; Checks that llvm.dbg.declare instructions are updated 
; accordingly as we merge allocas.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @_Z3zzzi(i32 %p) nounwind uwtable sanitize_address {
entry:
  %p.addr = alloca i32, align 4
  %r = alloca i32, align 4
  store i32 %p, i32* %p.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %p.addr}, metadata !10), !dbg !11
  call void @llvm.dbg.declare(metadata !{i32* %r}, metadata !12), !dbg !14
  %0 = load i32* %p.addr, align 4, !dbg !14
  %add = add nsw i32 %0, 1, !dbg !14
  store i32 %add, i32* %r, align 4, !dbg !14
  %1 = load i32* %r, align 4, !dbg !15
  ret i32 %1, !dbg !15
}

;   CHECK: define i32 @_Z3zzzi
;   CHECK: entry:
; Verify that llvm.dbg.declare calls are in the entry basic block.
;   CHECK-NOT: %entry
;   CHECK: call void @llvm.dbg.declare(metadata {{.*}}, metadata ![[ARG_ID:[0-9]+]])
;   CHECK-NOT: %entry
;   CHECK: call void @llvm.dbg.declare(metadata {{.*}}, metadata ![[VAR_ID:[0-9]+]])

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17}

!0 = metadata !{i32 786449, metadata !16, i32 4, metadata !"clang version 3.3 (trunk 169314)", i1 true, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1, null, metadata !""} ; [ DW_TAG_compile_unit ] [/usr/local/google/llvm_cmake_clang/tmp/debuginfo/a.cc] [DW_LANG_C_plus_plus]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 786478, metadata !16, metadata !6, metadata !"zzz", metadata !"zzz", metadata !"_Z3zzzi", i32 1, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @_Z3zzzi, null, null, metadata !1, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [zzz]
!6 = metadata !{i32 786473, metadata !16} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9, metadata !9}
!9 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{i32 786689, metadata !5, metadata !"p", metadata !6, i32 16777217, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [p] [line 1]
!11 = metadata !{i32 1, i32 0, metadata !5, null}
!12 = metadata !{i32 786688, metadata !13, metadata !"r", metadata !6, i32 2, metadata !9, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [r] [line 2]

; Verify that debug descriptors for argument and local variable will be replaced
; with descriptors that end with OpDeref (encoded as 2).
;   CHECK: ![[ARG_ID]] = metadata {{.*}} i64 2} ; [ DW_TAG_arg_variable ] [p] [line 1]
;   CHECK: ![[VAR_ID]] = metadata {{.*}} i64 2} ; [ DW_TAG_auto_variable ] [r] [line 2]
; Verify that there are no more variable descriptors.
;   CHECK-NOT: DW_TAG_arg_variable
;   CHECK-NOT: DW_TAG_auto_variable


!13 = metadata !{i32 786443, metadata !16, metadata !5, i32 1, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [/usr/local/google/llvm_cmake_clang/tmp/debuginfo/a.cc]
!14 = metadata !{i32 2, i32 0, metadata !13, null}
!15 = metadata !{i32 3, i32 0, metadata !13, null}
!16 = metadata !{metadata !"a.cc", metadata !"/usr/local/google/llvm_cmake_clang/tmp/debuginfo"}
!17 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
