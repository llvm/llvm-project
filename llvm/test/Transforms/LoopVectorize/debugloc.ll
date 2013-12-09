; RUN: opt -S < %s -loop-vectorize -force-vector-unroll=1 -force-vector-width=2 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Make sure we are preserving debug info in the vectorized code.

; CHECK: for.body.lr.ph
; CHECK:   cmp.zero = icmp eq i64 {{.*}}, 0, !dbg ![[LOC:[0-9]+]]
; CHECK: vector.body
; CHECK:   index {{.*}}, !dbg ![[LOC]]
; CHECK:   getelementptr inbounds i32* %a, {{.*}}, !dbg ![[LOC2:[0-9]+]]
; CHECK:   load <2 x i32>* {{.*}}, !dbg ![[LOC2]]
; CHECK:   add <2 x i32> {{.*}}, !dbg ![[LOC2]]
; CHECK:   add i64 %index, 2, !dbg ![[LOC]]
; CHECK:   icmp eq i64 %index.next, %end.idx.rnd.down, !dbg ![[LOC]]
; CHECK: middle.block
; CHECK:   add <2 x i32> %rdx.vec.exit.phi, %rdx.shuf, !dbg ![[LOC2]]
; CHECK:   extractelement <2 x i32> %bin.rdx, i32 0, !dbg ![[LOC2]]

define i32 @f(i32* nocapture %a, i32 %size) #0 {
entry:
  tail call void @llvm.dbg.value(metadata !{i32* %a}, i64 0, metadata !13), !dbg !19
  tail call void @llvm.dbg.value(metadata !{i32 %size}, i64 0, metadata !14), !dbg !19
  tail call void @llvm.dbg.value(metadata !2, i64 0, metadata !15), !dbg !20
  tail call void @llvm.dbg.value(metadata !2, i64 0, metadata !16), !dbg !21
  %cmp4 = icmp eq i32 %size, 0, !dbg !21
  br i1 %cmp4, label %for.end, label %for.body.lr.ph, !dbg !21

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body, !dbg !21

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %sum.05 = phi i32 [ 0, %for.body.lr.ph ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32* %a, i64 %indvars.iv, !dbg !22
  %0 = load i32* %arrayidx, align 4, !dbg !22
  %add = add i32 %0, %sum.05, !dbg !22
  tail call void @llvm.dbg.value(metadata !{i32 %add.lcssa}, i64 0, metadata !15), !dbg !22
  %indvars.iv.next = add i64 %indvars.iv, 1, !dbg !21
  tail call void @llvm.dbg.value(metadata !{null}, i64 0, metadata !16), !dbg !21
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !21
  %exitcond = icmp ne i32 %lftr.wideiv, %size, !dbg !21
  br i1 %exitcond, label %for.body, label %for.cond.for.end_crit_edge, !dbg !21

for.cond.for.end_crit_edge:                       ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  br label %for.end, !dbg !21

for.end:                                          ; preds = %entry, %for.cond.for.end_crit_edge
  %sum.0.lcssa = phi i32 [ %add.lcssa, %for.cond.for.end_crit_edge ], [ 0, %entry ]
  ret i32 %sum.0.lcssa, !dbg !26
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata) #1

attributes #0 = { nounwind readonly ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18, !27}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.4 (trunk 185038) (llvm/trunk 185097)", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/Volumes/Data/backedup/dev/os/llvm/debug/-] [DW_LANG_C99]
!1 = metadata !{metadata !"-", metadata !"/Volumes/Data/backedup/dev/os/llvm/debug"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"f", metadata !"f", metadata !"", i32 3, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (i32*, i32)* @f, null, null, metadata !12, i32 3} ; [ DW_TAG_subprogram ] [line 3] [def] [f]
!5 = metadata !{metadata !"<stdin>", metadata !"/Volumes/Data/backedup/dev/os/llvm/debug"}
!6 = metadata !{i32 786473, metadata !5}          ; [ DW_TAG_file_type ] [/Volumes/Data/backedup/dev/os/llvm/debug/<stdin>]
!7 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9, metadata !10, metadata !11}
!9 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!11 = metadata !{i32 786468, null, null, metadata !"unsigned int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ] [unsigned int] [line 0, size 32, align 32, offset 0, enc DW_ATE_unsigned]
!12 = metadata !{metadata !13, metadata !14, metadata !15, metadata !16}
!13 = metadata !{i32 786689, metadata !4, metadata !"a", metadata !6, i32 16777219, metadata !10, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [a] [line 3]
!14 = metadata !{i32 786689, metadata !4, metadata !"size", metadata !6, i32 33554435, metadata !11, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [size] [line 3]
!15 = metadata !{i32 786688, metadata !4, metadata !"sum", metadata !6, i32 4, metadata !11, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [sum] [line 4]
!16 = metadata !{i32 786688, metadata !17, metadata !"i", metadata !6, i32 5, metadata !11, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [i] [line 5]
!17 = metadata !{i32 786443, metadata !5, metadata !4, i32 5, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [/Volumes/Data/backedup/dev/os/llvm/debug/<stdin>]
!18 = metadata !{i32 2, metadata !"Dwarf Version", i32 3}
!19 = metadata !{i32 3, i32 0, metadata !4, null}
!20 = metadata !{i32 4, i32 0, metadata !4, null}
!21 = metadata !{i32 5, i32 0, metadata !17, null}
!22 = metadata !{i32 6, i32 0, metadata !17, null}
!26 = metadata !{i32 7, i32 0, metadata !4, null}
!27 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
