; RUN: opt < %s -scalarrepl -S | FileCheck %s
; RUN: opt < %s -scalarrepl-ssa -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.6.0"

; CHECK: f
; CHECK-NOT: llvm.dbg.declare
; CHECK: llvm.dbg.value
; CHECK: llvm.dbg.value
; CHECK: llvm.dbg.value
; CHECK: llvm.dbg.value
; CHECK: llvm.dbg.value

define i32 @f(i32 %a, i32 %b) nounwind ssp {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %c = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !6, metadata !{}), !dbg !7
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %b.addr, metadata !8, metadata !{}), !dbg !9
  call void @llvm.dbg.declare(metadata i32* %c, metadata !10, metadata !{}), !dbg !12
  %tmp = load i32* %a.addr, align 4, !dbg !13
  store i32 %tmp, i32* %c, align 4, !dbg !13
  %tmp1 = load i32* %a.addr, align 4, !dbg !14
  %tmp2 = load i32* %b.addr, align 4, !dbg !14
  %add = add nsw i32 %tmp1, %tmp2, !dbg !14
  store i32 %add, i32* %a.addr, align 4, !dbg !14
  %tmp3 = load i32* %c, align 4, !dbg !15
  %tmp4 = load i32* %b.addr, align 4, !dbg !15
  %sub = sub nsw i32 %tmp3, %tmp4, !dbg !15
  store i32 %sub, i32* %b.addr, align 4, !dbg !15
  %tmp5 = load i32* %a.addr, align 4, !dbg !16
  %tmp6 = load i32* %b.addr, align 4, !dbg !16
  %add7 = add nsw i32 %tmp5, %tmp6, !dbg !16
  ret i32 %add7, !dbg !16
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!20}

!0 = !{!"0x11\0012\00clang version 3.0 (trunk 131941)\000\00\000\00\000", !18, !19, !19, !17, null, null} ; [ DW_TAG_compile_unit ]
!1 = !{!"0x2e\00f\00f\00\001\000\001\000\006\00256\000\001", !18, !2, !3, null, i32 (i32, i32)* @f, null, null, null} ; [ DW_TAG_subprogram ] [line 1] [def] [f]
!2 = !{!"0x29", !18} ; [ DW_TAG_file_type ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !18, !2, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5}
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !0} ; [ DW_TAG_base_type ]
!6 = !{!"0x101\00a\0016777217\000", !1, !2, !5} ; [ DW_TAG_arg_variable ]
!7 = !MDLocation(line: 1, column: 11, scope: !1)
!8 = !{!"0x101\00b\0033554433\000", !1, !2, !5} ; [ DW_TAG_arg_variable ]
!9 = !MDLocation(line: 1, column: 18, scope: !1)
!10 = !{!"0x100\00c\002\000", !11, !2, !5} ; [ DW_TAG_auto_variable ]
!11 = !{!"0xb\001\0021\000", !18, !1} ; [ DW_TAG_lexical_block ]
!12 = !MDLocation(line: 2, column: 9, scope: !11)
!13 = !MDLocation(line: 2, column: 14, scope: !11)
!14 = !MDLocation(line: 3, column: 5, scope: !11)
!15 = !MDLocation(line: 4, column: 5, scope: !11)
!16 = !MDLocation(line: 5, column: 5, scope: !11)
!17 = !{!1}
!18 = !{!"/d/j/debug-test.c", !"/Volumes/Data/b"}
!19 = !{i32 0}
!20 = !{i32 1, !"Debug Info Version", i32 2}
