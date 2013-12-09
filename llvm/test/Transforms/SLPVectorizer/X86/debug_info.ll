; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.0"

; int depth(double *A, int m) {
;   double y0 = 0; double y1 = 1;
;   for (int i=0; i < m; i++) {
;     y0 = A[4];
;     y1 = A[5];
;   }
;   A[8] = y0; A[8+1] = y1;
; }

;CHECK: @depth
;CHECK: getelementptr inbounds {{.*}}, !dbg ![[LOC:[0-9]+]]
;CHECK: bitcast double* {{.*}}, !dbg ![[LOC]]
;CHECK: load <2 x double>* {{.*}}, !dbg ![[LOC]]
;CHECK: store <2 x double> {{.*}}, !dbg ![[LOC2:[0-9]+]]
;CHECK: ret
;CHECK: ![[LOC]] = metadata !{i32 4, i32 0,
;CHECK: ![[LOC2]] = metadata !{i32 7, i32 0,

define i32 @depth(double* nocapture %A, i32 %m) #0 {
entry:
  tail call void @llvm.dbg.value(metadata !{double* %A}, i64 0, metadata !12), !dbg !19
  tail call void @llvm.dbg.value(metadata !{i32 %m}, i64 0, metadata !13), !dbg !19
  tail call void @llvm.dbg.value(metadata !20, i64 0, metadata !14), !dbg !21
  tail call void @llvm.dbg.value(metadata !22, i64 0, metadata !15), !dbg !21
  tail call void @llvm.dbg.value(metadata !2, i64 0, metadata !16), !dbg !23
  %cmp8 = icmp sgt i32 %m, 0, !dbg !23
  br i1 %cmp8, label %for.body.lr.ph, label %for.end, !dbg !23

for.body.lr.ph:                                   ; preds = %entry
  %arrayidx = getelementptr inbounds double* %A, i64 4, !dbg !24
  %0 = load double* %arrayidx, align 8, !dbg !24
  %arrayidx1 = getelementptr inbounds double* %A, i64 5, !dbg !29
  %1 = load double* %arrayidx1, align 8, !dbg !29
  br label %for.end, !dbg !23

for.end:                                          ; preds = %for.body.lr.ph, %entry
  %y1.0.lcssa = phi double [ %1, %for.body.lr.ph ], [ 1.000000e+00, %entry ]
  %y0.0.lcssa = phi double [ %0, %for.body.lr.ph ], [ 0.000000e+00, %entry ]
  %arrayidx2 = getelementptr inbounds double* %A, i64 8, !dbg !30
  store double %y0.0.lcssa, double* %arrayidx2, align 8, !dbg !30
  %arrayidx3 = getelementptr inbounds double* %A, i64 9, !dbg !30
  store double %y1.0.lcssa, double* %arrayidx3, align 8, !dbg !30
  ret i32 undef, !dbg !31
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata) #1

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18, !32}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.4 (trunk 187335) (llvm/trunk 187335:187340M)", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/Users/nadav/file.c] [DW_LANG_C99]
!1 = metadata !{metadata !"file.c", metadata !"/Users/nadav"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"depth", metadata !"depth", metadata !"", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (double*, i32)* @depth, null, null, metadata !11, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [depth]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/Users/nadav/file.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !9, metadata !8}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !10} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from double]
!10 = metadata !{i32 786468, null, null, metadata !"double", i32 0, i64 64, i64 64, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ] [double] [line 0, size 64, align 64, offset 0, enc DW_ATE_float]
!11 = metadata !{metadata !12, metadata !13, metadata !14, metadata !15, metadata !16}
!12 = metadata !{i32 786689, metadata !4, metadata !"A", metadata !5, i32 16777217, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [A] [line 1]
!13 = metadata !{i32 786689, metadata !4, metadata !"m", metadata !5, i32 33554433, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [m] [line 1]
!14 = metadata !{i32 786688, metadata !4, metadata !"y0", metadata !5, i32 2, metadata !10, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [y0] [line 2]
!15 = metadata !{i32 786688, metadata !4, metadata !"y1", metadata !5, i32 2, metadata !10, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [y1] [line 2]
!16 = metadata !{i32 786688, metadata !17, metadata !"i", metadata !5, i32 3, metadata !8, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [i] [line 3]
!17 = metadata !{i32 786443, metadata !1, metadata !4, i32 3, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [/Users/nadav/file.c]
!18 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!19 = metadata !{i32 1, i32 0, metadata !4, null}
!20 = metadata !{double 0.000000e+00}
!21 = metadata !{i32 2, i32 0, metadata !4, null}
!22 = metadata !{double 1.000000e+00}
!23 = metadata !{i32 3, i32 0, metadata !17, null}
!24 = metadata !{i32 4, i32 0, metadata !25, null}
!25 = metadata !{i32 786443, metadata !1, metadata !17, i32 3, i32 0, i32 1} ; [ DW_TAG_lexical_block ] [/Users/nadav/file.c]
!29 = metadata !{i32 5, i32 0, metadata !25, null}
!30 = metadata !{i32 7, i32 0, metadata !4, null}
!31 = metadata !{i32 8, i32 0, metadata !4, null} ; [ DW_TAG_imported_declaration ]
!32 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
