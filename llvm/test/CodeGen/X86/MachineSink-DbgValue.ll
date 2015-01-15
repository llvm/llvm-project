; RUN: llc < %s | FileCheck %s
; Should sink matching DBG_VALUEs also.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

define i32 @foo(i32 %i, i32* nocapture %c) nounwind uwtable readonly ssp {
  tail call void @llvm.dbg.value(metadata i32 %i, i64 0, metadata !6, metadata !{!"0x102"}), !dbg !12
  %ab = load i32* %c, align 1, !dbg !14
  tail call void @llvm.dbg.value(metadata i32* %c, i64 0, metadata !7, metadata !{!"0x102"}), !dbg !13
  tail call void @llvm.dbg.value(metadata i32 %ab, i64 0, metadata !10, metadata !{!"0x102"}), !dbg !14
  %cd = icmp eq i32 %i, 42, !dbg !15
  br i1 %cd, label %bb1, label %bb2, !dbg !15

bb1:                                     ; preds = %0
;CHECK: DEBUG_VALUE: a
;CHECK:      .loc	1 5 5
;CHECK-NEXT: addl
  %gh = add nsw i32 %ab, 2, !dbg !16
  br label %bb2, !dbg !16

bb2:
  %.0 = phi i32 [ %gh, %bb1 ], [ 0, %0 ]
  ret i32 %.0, !dbg !17
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22}

!0 = !{!"0x11\0012\00Apple clang version 3.0 (tags/Apple/clang-211.10.1) (based on LLVM 3.0svn)\001\00\000\00\001", !20, !21, !21, !18, null,  null} ; [ DW_TAG_compile_unit ]
!1 = !{!"0x2e\00foo\00foo\00\002\000\001\000\006\00256\001\000", !20, !2, !3, null, i32 (i32, i32*)* @foo, null, null, !19} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 0] [foo]
!2 = !{!"0x29", !20} ; [ DW_TAG_file_type ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !20, !2, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5}
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !0} ; [ DW_TAG_base_type ]
!6 = !{!"0x101\00i\0016777218\000", !1, !2, !5} ; [ DW_TAG_arg_variable ]
!7 = !{!"0x101\00c\0033554434\000", !1, !2, !8} ; [ DW_TAG_arg_variable ]
!8 = !{!"0xf\00\000\0064\0064\000\000", null, !0, !9} ; [ DW_TAG_pointer_type ]
!9 = !{!"0x24\00char\000\008\008\000\000\006", null, !0} ; [ DW_TAG_base_type ]
!10 = !{!"0x100\00a\003\000", !11, !2, !9} ; [ DW_TAG_auto_variable ]
!11 = !{!"0xb\002\0025\000", !20, !1} ; [ DW_TAG_lexical_block ]
!12 = !MDLocation(line: 2, column: 13, scope: !1)
!13 = !MDLocation(line: 2, column: 22, scope: !1)
!14 = !MDLocation(line: 3, column: 14, scope: !11)
!15 = !MDLocation(line: 4, column: 3, scope: !11)
!16 = !MDLocation(line: 5, column: 5, scope: !11)
!17 = !MDLocation(line: 7, column: 1, scope: !11)
!18 = !{!1}
!19 = !{!6, !7, !10}
!20 = !{!"a.c", !"/private/tmp"}
!21 = !{i32 0}
!22 = !{i32 1, !"Debug Info Version", i32 2}
