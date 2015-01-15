; RUN: opt < %s -loop-vectorize -S -pass-remarks-missed='loop-vectorize' -pass-remarks-analysis='loop-vectorize' 2>&1 | FileCheck %s

; CHECK: remark: source.c:2:8: loop not vectorized: store that is conditionally executed prevents vectorization

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; Function Attrs: nounwind ssp uwtable
define void @conditional_store(i32* noalias nocapture %indices) #0 {
entry:
  br label %for.body, !dbg !10

for.body:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.inc ]
  %arrayidx = getelementptr inbounds i32* %indices, i64 %indvars.iv, !dbg !12
  %0 = load i32* %arrayidx, align 4, !dbg !12, !tbaa !14
  %cmp1 = icmp eq i32 %0, 1024, !dbg !12
  br i1 %cmp1, label %if.then, label %for.inc, !dbg !12

if.then:                                          ; preds = %for.body
  store i32 0, i32* %arrayidx, align 4, !dbg !18, !tbaa !14
  br label %for.inc, !dbg !18

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !10
  %exitcond = icmp eq i64 %indvars.iv.next, 4096, !dbg !10
  br i1 %exitcond, label %for.end, label %for.body, !dbg !10

for.end:                                          ; preds = %for.inc
  ret void, !dbg !19
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = !{!"0x11\0012\00clang version 3.6.0\001\00\000\00\002", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ]
!1 = !{!"source.c", !"."}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00conditional_store\00conditional_store\00\001\000\001\000\006\00256\001\001", !1, !5, !6, null, void (i32*)* @conditional_store, null, null, !2} ; [ DW_TAG_subprogram ]
!5 = !{!"0x29", !1} ; [ DW_TAG_file_type ]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !2, null, null, null} ; [ DW_TAG_subroutine_type ]
!7 = !{i32 2, !"Dwarf Version", i32 2}
!8 = !{i32 2, !"Debug Info Version", i32 2}
!9 = !{!"clang version 3.6.0"}
!10 = !MDLocation(line: 2, column: 8, scope: !11)
!11 = !{!"0xb\002\003\000", !1, !4} ; [ DW_TAG_lexical_block ]
!12 = !MDLocation(line: 3, column: 9, scope: !13)
!13 = !{!"0xb\003\009\000", !1, !11} ; [ DW_TAG_lexical_block ]
!14 = !{!15, !15, i64 0}
!15 = !{!"int", !16, i64 0}
!16 = !{!"omnipotent char", !17, i64 0}
!17 = !{!"Simple C/C++ TBAA"}
!18 = !MDLocation(line: 3, column: 29, scope: !13)
!19 = !MDLocation(line: 4, column: 1, scope: !4)
