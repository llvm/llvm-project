; RUN: opt -loop-idiom < %s -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"


define void @foo(double* nocapture %a) nounwind ssp {
entry:
  tail call void @llvm.dbg.value(metadata double* %a, i64 0, metadata !5, metadata !{}), !dbg !8
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !10, metadata !{}), !dbg !14
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %for.body ]
  %arrayidx = getelementptr double* %a, i64 %indvar
; CHECK: call void @llvm.memset{{.+}} !dbg 
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !15
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp ne i64 %indvar.next, 1000
  br i1 %exitcond, label %for.body, label %for.end, !dbg !14

for.end:                                          ; preds = %for.body
  tail call void @llvm.dbg.value(metadata !{null}, i64 0, metadata !10, metadata !{}), !dbg !16
  ret void, !dbg !17
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.module.flags = !{!19}
!llvm.dbg.sp = !{!0}

!0 = !{!"0x2e\00foo\00foo\00\002\000\001\000\006\00256\000\000", !18, !1, !3, null, void (double*)* @foo, null, null, null} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 0] [foo]
!1 = !{!"0x29", !18} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\0012\00clang version 2.9 (trunk 127165:127174)\001\00\000\00\000", !18, !9, !9, null, null, null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !18, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{null}
!5 = !{!"0x101\00a\0016777218\000", !0, !1, !6} ; [ DW_TAG_arg_variable ]
!6 = !{!"0xf\00\000\0064\0064\000\000", null, !2, !7} ; [ DW_TAG_pointer_type ]
!7 = !{!"0x24\00double\000\0064\0064\000\000\004", null, !2} ; [ DW_TAG_base_type ]
!8 = !MDLocation(line: 2, column: 18, scope: !0)
!9 = !{i32 0}
!10 = !{!"0x100\00i\003\000", !11, !1, !13} ; [ DW_TAG_auto_variable ]
!11 = !{!"0xb\003\003\001", !18, !12} ; [ DW_TAG_lexical_block ]
!12 = !{!"0xb\002\0021\000", !18, !0} ; [ DW_TAG_lexical_block ]
!13 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !2} ; [ DW_TAG_base_type ]
!14 = !MDLocation(line: 3, column: 3, scope: !12)
!15 = !MDLocation(line: 4, column: 5, scope: !11)
!16 = !MDLocation(line: 3, column: 29, scope: !11)
!17 = !MDLocation(line: 5, column: 1, scope: !12)
!18 = !{!"li.c", !"/private/tmp"}
!19 = !{i32 1, !"Debug Info Version", i32 2}
