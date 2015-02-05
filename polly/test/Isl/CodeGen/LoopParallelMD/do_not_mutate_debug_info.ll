; This test checks that we do not accidently mutate the debug info when
; inserting loop parallel metadata.
; RUN: opt %loadPolly < %s  -S -polly -polly-codegen-isl -polly-ast-detect-parallel | FileCheck %s
; CHECK-NOT: !7 = !{!7}
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global i32* null, align 8

; Function Attrs: nounwind uwtable
define void @foo() {
entry:
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !9, metadata !19), !dbg !20
  %0 = load i32** @A, align 8, !dbg !21, !tbaa !23
  br label %for.body, !dbg !27

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32* %0, i64 %indvars.iv, !dbg !21
  %1 = load i32* %arrayidx, align 4, !dbg !21, !tbaa !30
  %add = add nsw i32 %1, 1, !dbg !21
  store i32 %add, i32* %arrayidx, align 4, !dbg !21, !tbaa !30
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !27
  %exitcond = icmp eq i64 %indvars.iv, 1, !dbg !27
  br i1 %exitcond, label %for.end, label %for.body, !dbg !27

for.end:                                          ; preds = %for.body
  ret void, !dbg !32
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata)


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!15, !16}
!llvm.ident = !{!17}

!0 = !{!"0x11\0012\00clang version 3.6.0 \001\00\000\00\001", !1, !2, !2, !3, !12, !2} ; [ DW_TAG_compile_unit ] [/local/mnt/workspace/build/tip-Release/t2.c] [DW_LANG_C99]
!1 = !{!"t2.c", !"/local/mnt/workspace/build/tip-Release"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00foo\00foo\00\003\000\001\000\000\000\001\003", !1, !5, !6, null, void ()* @foo, null, null, !8} ; [ DW_TAG_subprogram ] [line 3] [def] [foo]
!5 = !{!"0x29", !1}    ; [ DW_TAG_file_type ] [/local/mnt/workspace/build/tip-Release/t2.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null}
!8 = !{!9}
!9 = !{!"0x100\00i\004\000", !10, !5, !11} ; [ DW_TAG_auto_variable ] [i] [line 4]
!10 = !{!"0xb\004\003\000", !1, !4} ; [ DW_TAG_lexical_block ] [/local/mnt/workspace/build/tip-Release/t2.c]
!11 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!12 = !{!13}
!13 = !{!"0x34\00A\00A\00\002\000\001", null, !5, !14, i32** @A, null} ; [ DW_TAG_variable ] [A] [line 2] [def]
!14 = !{!"0xf\00\000\0064\0064\000\000", null, null, !11} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 2}
!17 = !{!"clang version 3.6.0 "}
!18 = !{i32 0}
!19 = !{!"0x102"}               ; [ DW_TAG_expression ]
!20 = !MDLocation(line: 4, column: 12, scope: !10)
!21 = !MDLocation(line: 5, column: 5, scope: !22)
!22 = !{!"0xb\004\003\001", !1, !10} ; [ DW_TAG_lexical_block ] [/local/mnt/workspace/build/tip-Release/t2.c]
!23 = !{!24, !24, i64 0}
!24 = !{!"any pointer", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C/C++ TBAA"}
!27 = !MDLocation(line: 4, column: 3, scope: !28)
!28 = !{!"0xb\002", !1, !29} ; [ DW_TAG_lexical_block ] [/local/mnt/workspace/build/tip-Release/t2.c]
!29 = !{!"0xb\001", !1, !22} ; [ DW_TAG_lexical_block ] [/local/mnt/workspace/build/tip-Release/t2.c]
!30 = !{!31, !31, i64 0}
!31 = !{!"int", !25, i64 0}
!32 = !MDLocation(line: 6, column: 1, scope: !4)
