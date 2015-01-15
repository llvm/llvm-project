; RUN: opt < %s -asan -asan-module -asan-use-after-return=0 -S | FileCheck %s

; Checks that llvm.dbg.declare instructions are updated 
; accordingly as we merge allocas.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @_Z3zzzi(i32 %p) nounwind uwtable sanitize_address {
entry:
  %p.addr = alloca i32, align 4
  %r = alloca i32, align 4
  store i32 %p, i32* %p.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %p.addr, metadata !10, metadata !{!"0x102"}), !dbg !11
  call void @llvm.dbg.declare(metadata i32* %r, metadata !12, metadata !{!"0x102"}), !dbg !14
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
;   CHECK: call void @llvm.dbg.declare(metadata {{.*}}, metadata ![[ARG_ID:[0-9]+]], metadata ![[OPDEREF:[0-9]+]])
;   CHECK-NOT: %entry
;   CHECK: call void @llvm.dbg.declare(metadata {{.*}}, metadata ![[VAR_ID:[0-9]+]], metadata ![[OPDEREF:[0-9]+]])

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17}

!0 = !{!"0x11\004\00clang version 3.3 (trunk 169314)\001\00\000\00\000", !16, !1, !1, !3, !1, null} ; [ DW_TAG_compile_unit ] [/usr/local/google/llvm_cmake_clang/tmp/debuginfo/a.cc] [DW_LANG_C_plus_plus]
!1 = !{i32 0}
!3 = !{!5}
!5 = !{!"0x2e\00zzz\00zzz\00_Z3zzzi\001\000\001\000\006\00256\000\001", !16, !6, !7, null, i32 (i32)* @_Z3zzzi, null, null, !1} ; [ DW_TAG_subprogram ] [line 1] [def] [zzz]
!6 = !{!"0x29", !16} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9, !9}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = !{!"0x101\00p\0016777217\000", !5, !6, !9} ; [ DW_TAG_arg_variable ] [p] [line 1]
!11 = !MDLocation(line: 1, scope: !5)
!12 = !{!"0x100\00r\002\000", !13, !6, !9} ; [ DW_TAG_auto_variable ] [r] [line 2]

; Verify that debug descriptors for argument and local variable will be replaced
; with descriptors that end with OpDeref (encoded as 2).
;   CHECK: ![[ARG_ID]] = {{.*}} ; [ DW_TAG_arg_variable ] [p] [line 1]
;   CHECK: ![[OPDEREF]] = !{!"0x102\006"}
;   CHECK: ![[VAR_ID]] = {{.*}} ; [ DW_TAG_auto_variable ] [r] [line 2]
; Verify that there are no more variable descriptors.
;   CHECK-NOT: DW_TAG_arg_variable
;   CHECK-NOT: DW_TAG_auto_variable


!13 = !{!"0xb\001\000\000", !16, !5} ; [ DW_TAG_lexical_block ] [/usr/local/google/llvm_cmake_clang/tmp/debuginfo/a.cc]
!14 = !MDLocation(line: 2, scope: !13)
!15 = !MDLocation(line: 3, scope: !13)
!16 = !{!"a.cc", !"/usr/local/google/llvm_cmake_clang/tmp/debuginfo"}
!17 = !{i32 1, !"Debug Info Version", i32 2}
