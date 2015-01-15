; RUN: llc < %s -mtriple=xcore-unknown-unknown -O0 | FileCheck %s

; target datalayout = "e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i64:32-f64:32-a:0:32-n32"
; target triple = "xcore"

; CHECK-LABEL: f
; CHECK: entsp 2
; ...the prologue...
; CHECK: .loc 1 2 0 prologue_end      # :2:0
; CHECK: add r0, r0, 1
; CHECK: retsp 2
define i32 @f(i32 %a) {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !11, metadata !{!"0x102"}), !dbg !12
  %0 = load i32* %a.addr, align 4, !dbg !12
  %add = add nsw i32 %0, 1, !dbg !12
  ret i32 %add, !dbg !12
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!0 = !{!"0x11\0012\00\000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ]
!1 = !{!"", !""}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00f\00f\00\002\000\001\000\006\00256\000\002", !1, !5, !6, null, i32 (i32)* @f, null, null, !2} ; [ DW_TAG_subprogram ]
!5 = !{!"0x29", !1} ; [ DW_TAG_file_type ]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ]
!7 = !{!8, !8}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 2}
!11 = !{!"0x101\00a\0016777218\000", !4, !5, !8} ; [ DW_TAG_arg_variable ]
!12 = !MDLocation(line: 2, scope: !4)

